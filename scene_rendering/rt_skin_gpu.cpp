#include "rt_skin_gpu.h"
#include "renderer/renderer_helper.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <iostream>
#include <chrono>
namespace engine::scene_rendering {
namespace er=renderer;
namespace {
struct Vertex { glm::vec4 p; glm::uvec4 j0; glm::vec4 w0; glm::uvec4 j1; glm::vec4 w1; };
struct Job { glm::mat4 model; glm::vec4 shape; glm::uvec4 src,dst; };
static_assert(sizeof(Vertex)==80 && sizeof(Job)==112);
template<class T> bool same(const std::vector<T>& a,const std::vector<T>* b) {
    return b ? a.size()==b->size() && (a.empty() || std::memcmp(a.data(),b->data(),a.size()*sizeof(T))==0) : a.empty();
}
template<class T> void snapshot(std::vector<T>& a,const std::vector<T>* b) { if(b) a=*b; else a.clear(); }
}
struct RtSkinGpu::Impl {
    std::shared_ptr<er::Device> device;
    std::shared_ptr<er::DescriptorSetLayout> layout;
    std::shared_ptr<er::PipelineLayout> pipeline_layout;
    std::shared_ptr<er::Pipeline> pipeline;
    std::shared_ptr<er::Queue> queue;
    std::shared_ptr<er::CommandPool> command_pool;
    struct Frame {
        std::array<er::BufferInfo,5> inputs;
        std::array<uint64_t,5> capacities{};
        std::shared_ptr<er::DescriptorSet> descriptor;
        std::shared_ptr<er::CommandBuffer> command;
        std::shared_ptr<er::Semaphore> ready;
        std::shared_ptr<er::QueryPool> timing;
        bool timing_submitted = false;
        uint32_t timing_jobs = 0, timing_vertices = 0;
        uint32_t revision=0;
    };
    std::array<Frame,2> frames;
    struct Mesh {
        std::vector<glm::vec3> p;
        std::vector<uint32_t> i;
        std::vector<glm::u16vec4> j0,j1;
        std::vector<glm::vec4> w0,w1;
        uint32_t vertex=0,index=0;
    };
    using Key=std::array<uintptr_t,7>;
    std::map<Key,Mesh> meshes;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Job> jobs;
    std::vector<glm::mat4> palette;
    std::vector<glm::uvec2> tasks;
    std::vector<uint32_t> topology;
    uint32_t revision=1,slot=0,nvertices=0,nindices=0;
    bool pending=false,changed=true,reported_workload=false;
    std::chrono::steady_clock::time_point last_timing_log{};
    std::vector<Job> last_jobs;
    std::vector<glm::mat4> last_palette;
    uint32_t last_revision=0;
    void upload(Frame& f,int binding,uint64_t size,const void* data) {
        uint64_t need=std::max(uint64_t(16),size);
        if (need>f.capacities[binding]) {
            f.inputs[binding].destroy(device);
            uint64_t cap=256;while(cap<need) cap*=2;
            er::Helper::createBuffer(device,SET_FLAG_BIT(BufferUsage,STORAGE_BUFFER_BIT),
                SET_2_FLAG_BITS(MemoryProperty,HOST_VISIBLE_BIT,HOST_COHERENT_BIT),0,
                f.inputs[binding].buffer,f.inputs[binding].memory,std::source_location::current(),cap,nullptr);
            f.capacities[binding]=cap;
        }
        // Inputs (including bone matrices) belong to the retired FIF slot.
        // Write directly: deferring would copy the palette again and couple
        // this independent slot to the previous frame's shared-buffer flush.
        if(size) device->updateBufferMemory(f.inputs[binding].memory,size,data,0,false);
    }
};
RtSkinGpu::RtSkinGpu(const std::shared_ptr<er::Device>& device,const std::shared_ptr<er::DescriptorPool>& pool)
    : impl_(std::make_unique<Impl>()) {
    auto& s=*impl_;s.device=device;
    std::vector<er::DescriptorSetLayoutBinding> bindings(9);
    for(uint32_t i=0;i<9;++i) bindings[i]=er::helper::getBufferDescriptionSetLayoutBinding(
        i,SET_FLAG_BIT(ShaderStage,COMPUTE_BIT),er::DescriptorType::STORAGE_BUFFER);
    s.layout=device->createDescriptorSetLayout(bindings);
    s.pipeline_layout=er::helper::createComputePipelineLayout(device,{s.layout},16);
    s.pipeline=er::helper::createComputePipeline(device,s.pipeline_layout,"rt_skin_comp.spv",std::source_location::current());
    s.queue=device->getAsyncComputeQueue();
    auto sets=device->createDescriptorSets(pool,s.layout,2);
    if(s.queue) s.command_pool=device->createCommandPool(device->getAsyncComputeFamily(),
        static_cast<uint32_t>(er::CommandPoolCreateFlagBits::RESET_COMMAND_BUFFER_BIT));
    for(uint32_t i=0;i<2;++i) {
        s.frames[i].descriptor=sets[i];
        if(s.queue) {
            s.frames[i].command=device->allocateCommandBuffers(s.command_pool,1,true)[0];
            s.frames[i].ready=device->createSemaphore(std::source_location::current());
            s.frames[i].timing=device->createQueryPool(5);
        }
    }
    std::cout<<"[RT_SKIN] GPU skinning + bounds: "<<(s.queue?"async queue":"graphics queue fallback")<<std::endl;
}
RtSkinGpu::~RtSkinGpu() {
    auto& s=*impl_;
    for(auto& f:s.frames) {
        for(auto& b:f.inputs) b.destroy(s.device);
        if(f.ready) s.device->destroySemaphore(f.ready);
        if(f.timing) s.device->destroyQueryPool(f.timing);
    }
    if(s.command_pool) s.device->destroyCommandPool(s.command_pool);
    s.device->destroyPipeline(s.pipeline);
    s.device->destroyPipelineLayout(s.pipeline_layout);
    s.device->destroyDescriptorSetLayout(s.layout);
}
void RtSkinGpu::prepare(const std::vector<RtSkinBatch>& batches,uint32_t slot) {
    auto& s=*impl_; if(s.pending) throw std::logic_error("RT skin compute was not submitted");
    s.slot=slot%2;s.jobs.clear();s.palette.clear();s.tasks.clear();s.topology.clear();s.nvertices=s.nindices=0;
    // The caller retired this slot's graphics fence. Graphics waits for
    // this slot's compute semaphore, so its queries are also safe to read.
    // Read without WAIT; no extra queue/device/fence synchronization.
    auto& retired = s.frames[s.slot];
    if (retired.timing_submitted) {
        std::vector<uint64_t> ticks;
        if (s.device->getQueryPoolResults(retired.timing, 0, 5, ticks) && ticks.size() == 5) {
            const auto now = std::chrono::steady_clock::now();
            if (now - s.last_timing_log >= std::chrono::seconds(1)) {
                const double ms = s.device->getTimestampPeriod() * 1e-6;
                std::cout << "[RT_SKIN_GPU] async retired slot=" << s.slot
                    << " instances=" << retired.timing_jobs
                    << " vertices=" << retired.timing_vertices
                    << " deform_ms=" << (ticks[1]-ticks[0])*ms
                    << " chunk_bounds_ms=" << (ticks[2]-ticks[1])*ms
                    << " header_barriers_ms=" << (ticks[3]-ticks[2])*ms
                    << " acceleration_structures_ms=" << (ticks[4]-ticks[3])*ms
                    << " total_ms=" << (ticks[4]-ticks[0])*ms
                    << " (queue-local elapsed; may overlap graphics)" << std::endl;
                s.last_timing_log = now;
            }
        }
        retired.timing_submitted = false;
    }
    std::set<Impl::Key> checked;
    for(const auto& b:batches) {
        if(!b.positions || !b.indices || b.positions->empty() || b.indices->size()<3) continue;
        const auto* jm=b.joint_matrices?b.joint_matrices:&b.palette;
        uint32_t mode=b.world_space_dynamic?0u:b.deformation;
        if(b.joints && b.weights) {
            if(jm->empty()) continue;
            mode=b.deformation==3?3u:1u;
        }
        if((mode==1 || mode==3) && (!b.joints || !b.weights || b.joints->size()!=b.positions->size() || b.weights->size()!=b.positions->size())) continue;
        if(mode==2 && jm->size()<3) continue;
        Impl::Key key{reinterpret_cast<uintptr_t>(b.positions),reinterpret_cast<uintptr_t>(b.indices),
            reinterpret_cast<uintptr_t>(b.joints),reinterpret_cast<uintptr_t>(b.weights),
            reinterpret_cast<uintptr_t>(b.joints1),reinterpret_cast<uintptr_t>(b.weights1),0};
        auto& mesh=s.meshes[key];
        if(checked.insert(key).second && (!same(mesh.p,b.positions) || !same(mesh.i,b.indices) ||
            !same(mesh.j0,b.joints) || !same(mesh.w0,b.weights) || !same(mesh.j1,b.joints1) || !same(mesh.w1,b.weights1))) {
            snapshot(mesh.p,b.positions);snapshot(mesh.i,b.indices);snapshot(mesh.j0,b.joints);snapshot(mesh.w0,b.weights);
            snapshot(mesh.j1,b.joints1);snapshot(mesh.w1,b.weights1);
            mesh.vertex=uint32_t(s.vertices.size());mesh.index=uint32_t(s.indices.size());
            for(size_t v=0;v<mesh.p.size();++v) {
                Vertex x{};x.p=glm::vec4(mesh.p[v],1);
                if(mesh.j0.size()==mesh.p.size() && mesh.w0.size()==mesh.p.size()) {x.j0=mesh.j0[v];x.w0=mesh.w0[v];}
                if(mesh.j1.size()==mesh.p.size() && mesh.w1.size()==mesh.p.size()) {x.j1=mesh.j1[v];x.w1=mesh.w1[v];}
                s.vertices.push_back(x);
            }
            s.indices.insert(s.indices.end(),mesh.i.begin(),mesh.i.end());++s.revision;
        }
        const uint32_t nv=uint32_t(b.positions->size()),ni=uint32_t(b.indices->size()/3)*3;
        Job job{b.world_space_dynamic?glm::mat4(1):b.model,b.shape,
            glm::uvec4(mesh.vertex,nv,mesh.index,ni),
            glm::uvec4(s.nvertices,s.nindices,uint32_t(s.palette.size()),(mode<<24)|uint32_t(jm->size()))};
        const uint32_t id=uint32_t(s.jobs.size());s.jobs.push_back(job);
        s.palette.insert(s.palette.end(),jm->begin(),jm->end());
        for(uint32_t k=0;k<(std::max(nv,ni)+63)/64;++k) s.tasks.emplace_back(id,k);
        s.topology.insert(s.topology.end(),{mesh.index,nv,ni});
        s.nvertices+=nv;s.nindices+=ni;
    }
    if(s.nvertices && !s.reported_workload) {
        std::cout<<"[RT_SKIN] first GPU workload: "<<s.jobs.size()<<" instances, "
                 <<s.nvertices<<" vertices, "<<s.nindices/3<<" triangles"<<std::endl;
        s.reported_workload=true;
    }
    s.changed = s.last_revision!=s.revision || !same(s.last_jobs,&s.jobs) || !same(s.last_palette,&s.palette);
    s.last_jobs=s.jobs;s.last_palette=s.palette;s.last_revision=s.revision;
    auto& f=s.frames[s.slot];
    if(f.revision!=s.revision) {
        s.upload(f,0,s.vertices.size()*sizeof(Vertex),s.vertices.data());
        s.upload(f,1,s.indices.size()*sizeof(uint32_t),s.indices.data());f.revision=s.revision;
    }
    s.upload(f,2,s.jobs.size()*sizeof(Job),s.jobs.data());
    s.upload(f,3,s.palette.size()*sizeof(glm::mat4),s.palette.data());
    s.upload(f,4,s.tasks.size()*sizeof(glm::uvec2),s.tasks.data());
}
bool RtSkinGpu::pending() const {return impl_->pending;}
bool RtSkinGpu::changed() const {return impl_->changed;}
uint32_t RtSkinGpu::vertices() const {return impl_->nvertices;}
uint32_t RtSkinGpu::indices() const {return impl_->nindices;}
uint32_t RtSkinGpu::chunks() const {return (impl_->nindices/3+127)/128;}
const std::vector<uint32_t>& RtSkinGpu::topology() const {return impl_->topology;}
std::shared_ptr<er::CommandBuffer> RtSkinGpu::record(const std::shared_ptr<er::CommandBuffer>& graphics,
    er::BufferInfo& positions,er::BufferInfo& indices,er::BufferInfo& chunks_buffer,er::BufferInfo& headers) {
    auto& s=*impl_;auto& f=s.frames[s.slot];auto cmd=s.queue?f.command:graphics;
    if(s.queue) {
        cmd->reset(0);cmd->beginCommandBuffer(0);
        cmd->resetQueryPool(f.timing,0,5);
        cmd->writeTimestamp(f.timing,0,false);
        f.timing_jobs=uint32_t(s.jobs.size());f.timing_vertices=s.nvertices;
    }
    er::WriteDescriptorList writes;
    for(uint32_t i=0;i<5;++i) er::Helper::addOneBuffer(writes,f.descriptor,er::DescriptorType::STORAGE_BUFFER,i,f.inputs[i].buffer,uint32_t(f.capacities[i]));
    const er::BufferInfo* outputs[]={&positions,&indices,&chunks_buffer,&headers};
    const uint32_t sizes[]={std::max(16u,vertices()*16u),std::max(16u,this->indices()*4u),std::max(32u,chunks()*32u),64u};
    for(uint32_t i=0;i<4;++i) er::Helper::addOneBuffer(writes,f.descriptor,er::DescriptorType::STORAGE_BUFFER,5+i,outputs[i]->buffer,sizes[i]);
    s.device->updateDescriptorSets(writes);
    cmd->beginDebugUtilsLabel("RT GPU skinning and bounds");
    cmd->bindPipeline(er::PipelineBindPoint::COMPUTE,s.pipeline);
    cmd->bindDescriptorSets(er::PipelineBindPoint::COMPUTE,s.pipeline_layout,{f.descriptor});
    auto dispatch=[&](uint32_t phase,uint32_t n) {
        for(uint32_t base=0;base<n;base+=65535) {
            glm::uvec4 pc(phase,base,this->indices(),chunks());
            cmd->pushConstants(SET_FLAG_BIT(ShaderStage,COMPUTE_BIT),s.pipeline_layout,&pc,sizeof(pc));
            cmd->dispatch(std::min(65535u,n-base),1,1);
        }
    };
    er::BufferResourceInfo write{SET_FLAG_BIT(Access,SHADER_WRITE_BIT),SET_FLAG_BIT(PipelineStage,COMPUTE_SHADER_BIT)};
    er::BufferResourceInfo read{SET_FLAG_BIT(Access,SHADER_READ_BIT),SET_FLAG_BIT(PipelineStage,COMPUTE_SHADER_BIT)};
    dispatch(0,uint32_t(s.tasks.size()));
    if(s.queue) cmd->writeTimestamp(f.timing,1,true);
    cmd->addBufferBarrier(positions.buffer,write,read);cmd->addBufferBarrier(indices.buffer,write,read);
    dispatch(1,chunks());cmd->addBufferBarrier(chunks_buffer.buffer,write,read);
    if(s.queue) cmd->writeTimestamp(f.timing,2,true);
    dispatch(2,1);
    er::BufferResourceInfo consumers{
        SET_FLAG_BIT(Access,SHADER_READ_BIT)|SET_FLAG_BIT(Access,ACCELERATION_STRUCTURE_READ_BIT_KHR),
        SET_FLAG_BIT(PipelineStage,COMPUTE_SHADER_BIT)|SET_FLAG_BIT(PipelineStage,ACCELERATION_STRUCTURE_BUILD_BIT_KHR)};
    for(const auto* b:outputs) cmd->addBufferBarrier(b->buffer,write,consumers);
    if(s.queue) cmd->writeTimestamp(f.timing,3,true);
    cmd->endDebugUtilsLabel();
    return cmd;
}
void RtSkinGpu::finish() {
    auto& s=*impl_;
    if(s.queue) {
        auto& f=s.frames[s.slot];
        f.command->writeTimestamp(f.timing,4,true);
        f.command->endCommandBuffer();s.pending=true;
    }
}
std::shared_ptr<er::Semaphore> RtSkinGpu::submit() {
    auto& s=*impl_;if(!s.pending) return {};
    auto& f=s.frames[s.slot];er::Helper::submitQueue(s.queue,{}, {},{f.command},{f.ready},{});
    f.timing_submitted=true;
    s.pending=false;return f.ready;
}
}
