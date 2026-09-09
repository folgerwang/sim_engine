// Standalone Vulkan readback regression for rt_skin.comp; no window or scene.
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <array>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <algorithm>
#define VK_CHECK(x) do {auto r=(x);if(r!=VK_SUCCESS) throw std::runtime_error(#x);} while(0)
struct Vertex {glm::vec4 p;glm::uvec4 j0;glm::vec4 w0;glm::uvec4 j1;glm::vec4 w1;};
struct Job {glm::mat4 model;glm::vec4 shape;glm::uvec4 src,dst;};
struct Header {glm::vec4 lo,hi;glm::uvec4 range;};
struct Buffer {VkBuffer b{};VkDeviceMemory m{};void* p{};};
int main(int argc,char** argv) try {
    if(argc<2) throw std::runtime_error("usage: RtSkinShaderTest path/to/rt_skin_comp.spv");
    VkInstance instance;VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};app.apiVersion=VK_API_VERSION_1_3;ici.pApplicationInfo=&app;
    VK_CHECK(vkCreateInstance(&ici,nullptr,&instance));
    uint32_t n=0;VK_CHECK(vkEnumeratePhysicalDevices(instance,&n,nullptr));
    std::vector<VkPhysicalDevice> devices(n);VK_CHECK(vkEnumeratePhysicalDevices(instance,&n,devices.data()));
    if(!n) throw std::runtime_error("no Vulkan device");auto physical=devices[0];
    uint32_t nf=0;vkGetPhysicalDeviceQueueFamilyProperties(physical,&nf,nullptr);
    std::vector<VkQueueFamilyProperties> families(nf);vkGetPhysicalDeviceQueueFamilyProperties(physical,&nf,families.data());
    uint32_t family=0;while(family<nf && !(families[family].queueFlags&VK_QUEUE_COMPUTE_BIT)) ++family;
    if(family==nf) throw std::runtime_error("no compute family");
    const uint32_t nq=std::min(2u,families[family].queueCount);float priorities[]={1,1};
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};qci.queueFamilyIndex=family;qci.queueCount=nq;qci.pQueuePriorities=priorities;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};dci.queueCreateInfoCount=1;dci.pQueueCreateInfos=&qci;
    VkDevice device;VK_CHECK(vkCreateDevice(physical,&dci,nullptr,&device));
    VkQueue queue,async;vkGetDeviceQueue(device,family,0,&queue);vkGetDeviceQueue(device,family,nq-1,&async);
    VkPhysicalDeviceMemoryProperties mem;vkGetPhysicalDeviceMemoryProperties(physical,&mem);
    std::array<Buffer,9> buffers;
    constexpr size_t capacity=8*1024*1024;
    for(auto& b:buffers) {
        VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};ci.size=capacity;ci.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        VK_CHECK(vkCreateBuffer(device,&ci,nullptr,&b.b));VkMemoryRequirements req;vkGetBufferMemoryRequirements(device,b.b,&req);
        uint32_t type=0;for(;type<mem.memoryTypeCount;++type) if((req.memoryTypeBits&(1u<<type)) &&
            (mem.memoryTypes[type].propertyFlags&(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))==(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) break;
        if(type==mem.memoryTypeCount) throw std::runtime_error("no coherent memory");
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.allocationSize=req.size;ai.memoryTypeIndex=type;
        VK_CHECK(vkAllocateMemory(device,&ai,nullptr,&b.m));VK_CHECK(vkBindBufferMemory(device,b.b,b.m,0));VK_CHECK(vkMapMemory(device,b.m,0,capacity,0,&b.p));
    }
    std::array<VkDescriptorSetLayoutBinding,9> bindings{};
    for(uint32_t i=0;i<9;++i) bindings[i]={i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};lci.bindingCount=9;lci.pBindings=bindings.data();
    VkDescriptorSetLayout layout;VK_CHECK(vkCreateDescriptorSetLayout(device,&lci,nullptr,&layout));
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,9};VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};pci.maxSets=1;pci.poolSizeCount=1;pci.pPoolSizes=&ps;
    VkDescriptorPool pool;VK_CHECK(vkCreateDescriptorPool(device,&pci,nullptr,&pool));
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};dai.descriptorPool=pool;dai.descriptorSetCount=1;dai.pSetLayouts=&layout;
    VkDescriptorSet set;VK_CHECK(vkAllocateDescriptorSets(device,&dai,&set));
    std::array<VkDescriptorBufferInfo,9> bi;std::array<VkWriteDescriptorSet,9> writes{};
    for(uint32_t i=0;i<9;++i){bi[i]={buffers[i].b,0,capacity};writes[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;writes[i].dstSet=set;writes[i].dstBinding=i;writes[i].descriptorCount=1;writes[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;writes[i].pBufferInfo=&bi[i];}
    vkUpdateDescriptorSets(device,9,writes.data(),0,nullptr);
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT,0,16};VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};plci.setLayoutCount=1;plci.pSetLayouts=&layout;plci.pushConstantRangeCount=1;plci.pPushConstantRanges=&range;
    VkPipelineLayout pl;VK_CHECK(vkCreatePipelineLayout(device,&plci,nullptr,&pl));
    std::ifstream file(argv[1],std::ios::binary|std::ios::ate);if(!file) throw std::runtime_error("shader missing");size_t bytes=size_t(file.tellg());std::vector<uint32_t> code(bytes/4);file.seekg(0);file.read(reinterpret_cast<char*>(code.data()),bytes);
    VkShaderModuleCreateInfo sci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};sci.codeSize=bytes;sci.pCode=code.data();VkShaderModule shader;VK_CHECK(vkCreateShaderModule(device,&sci,nullptr,&shader));
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};cpi.layout=pl;cpi.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;cpi.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;cpi.stage.module=shader;cpi.stage.pName="main";
    VkPipeline pipeline;VK_CHECK(vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&cpi,nullptr,&pipeline));
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};cpci.queueFamilyIndex=family;cpci.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;VkCommandPool cp;VK_CHECK(vkCreateCommandPool(device,&cpci,nullptr,&cp));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};cai.commandPool=cp;cai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;cai.commandBufferCount=2;VkCommandBuffer cmd[2];VK_CHECK(vkAllocateCommandBuffers(device,&cai,cmd));
    VkSemaphoreCreateInfo semci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};VkSemaphore ready;VK_CHECK(vkCreateSemaphore(device,&semci,nullptr,&ready));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};VkFence fence;VK_CHECK(vkCreateFence(device,&fci,nullptr,&fence));
    for(uint32_t run=0;run<4;++run) {
        // Rigid, eight-weight normalized skinning (including invalid joint
        // indices and zero weights), and citizen tapered/pivot-blended parts.
        uint32_t instances=run==2?0u:(run==3?10000u:4u);
        std::vector<Vertex> verts(3);
        verts[0].p={-1,-1,0,1};verts[1].p={1,0,0,1};verts[2].p={0,1,1,1};
        verts[0].j0={0,1,99,0};verts[0].w0={2,1,1,0};verts[0].j1={1,0,0,0};verts[0].w1={4,0,0,0};
        verts[1].w0={0,0,0,0};verts[2].j0={0,0,0,0};verts[2].w0={1,0,0,0};
        std::vector<uint32_t> idx{0,1,2};std::vector<glm::mat4> palettes{glm::translate(glm::mat4(1),glm::vec3(2,3,4)),glm::translate(glm::mat4(1),glm::vec3(-2,1,5)),glm::mat4(1)};
        std::vector<Job> jobs;std::vector<glm::uvec2> tasks;std::vector<glm::vec3> expected;
        for(uint32_t j=0;j<instances;++j) {
            uint32_t mode=j%4;Job job{glm::translate(glm::mat4(1),glm::vec3(float(j),0,0)),glm::vec4(.2f,-.3f,.4f,.65f),glm::uvec4(0,3,0,3),glm::uvec4(j*3,j*3,0,(mode<<24)|3u)};jobs.push_back(job);tasks.emplace_back(j,0);
            for(const auto& v:verts) {
                glm::vec4 p=v.p;
                if(mode==1 || mode==3) {glm::mat4 m(0);float sum=0;for(int k=0;k<4;++k){m+=v.w0[k]*palettes[std::min(v.j0[k],2u)]+v.w1[k]*palettes[std::min(v.j1[k],2u)];sum+=v.w0[k]+v.w1[k];}if(sum>1e-4f)p=m/sum*p;else if(mode==3)p=palettes[0]*p;}
                if(mode==2) {float f=glm::mix(job.shape.w,1.f,(p.y+1)*.5f);p.x*=f;p.z*=f;float wu=glm::clamp(.5f+(v.p.y-job.shape.x)/(2*job.shape.z),0.f,1.f),wl=glm::clamp(.5f-(v.p.y-job.shape.y)/(2*job.shape.z),0.f,1.f);p=std::max(0.f,1-wu-wl)*(palettes[0]*p)+wu*(palettes[1]*p)+wl*(palettes[2]*p);}
                expected.push_back(glm::vec3(job.model*p));
            }
        }
        auto upload=[&](int i,const auto& v){if(v.size()*sizeof(v[0])>capacity) throw std::runtime_error("fixture too big");if(!v.empty())std::memcpy(buffers[i].p,v.data(),v.size()*sizeof(v[0]));};
        upload(0,verts);upload(1,idx);upload(2,jobs);upload(3,palettes);upload(4,tasks);
        for(auto c:cmd){VK_CHECK(vkResetCommandBuffer(c,0));VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};VK_CHECK(vkBeginCommandBuffer(c,&begin));}
        vkCmdBindPipeline(cmd[0],VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);vkCmdBindDescriptorSets(cmd[0],VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&set,0,nullptr);
        uint32_t chunks=(instances+127)/128;
        auto dispatch=[&](uint32_t phase,uint32_t n){glm::uvec4 pc(phase,0,instances*3,chunks);vkCmdPushConstants(cmd[0],pl,VK_SHADER_STAGE_COMPUTE_BIT,0,16,&pc);if(n)vkCmdDispatch(cmd[0],n,1,1);};
        auto barrier=[&](){VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};b.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;b.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;vkCmdPipelineBarrier(cmd[0],VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&b,0,nullptr,0,nullptr);};
        dispatch(0,instances);barrier();dispatch(1,chunks);barrier();dispatch(2,1);
        VK_CHECK(vkEndCommandBuffer(cmd[0]));
        VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};host.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;host.dstAccessMask=VK_ACCESS_HOST_READ_BIT;vkCmdPipelineBarrier(cmd[1],VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&host,0,nullptr,0,nullptr);VK_CHECK(vkEndCommandBuffer(cmd[1]));
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};submit.commandBufferCount=1;submit.pCommandBuffers=&cmd[0];submit.signalSemaphoreCount=1;submit.pSignalSemaphores=&ready;VK_CHECK(vkQueueSubmit(run%2?async:queue,1,&submit,VK_NULL_HANDLE));
        VkPipelineStageFlags wait=VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;submit.pCommandBuffers=&cmd[1];submit.waitSemaphoreCount=1;submit.pWaitSemaphores=&ready;submit.pWaitDstStageMask=&wait;submit.signalSemaphoreCount=0;
        VK_CHECK(vkQueueSubmit(queue,1,&submit,fence));VK_CHECK(vkWaitForFences(device,1,&fence,VK_TRUE,UINT64_MAX));VK_CHECK(vkResetFences(device,1,&fence));
        auto output=static_cast<glm::vec4*>(buffers[5].p);auto output_idx=static_cast<uint32_t*>(buffers[6].p);
        for(size_t i=0;i<expected.size();++i){if(glm::length(glm::vec3(output[i])-expected[i])>0.002f || output_idx[i]!=i) throw std::runtime_error("vertex/index mismatch");}
        auto count=static_cast<glm::uvec4*>(buffers[8].p);auto h=reinterpret_cast<Header*>(count+1);
        if(count->x!=(instances?1u:0u) || h->range.y!=instances) throw std::runtime_error("header mismatch");
        auto bounds=static_cast<glm::vec4*>(buffers[7].p);
        for(uint32_t c=0;c<chunks;++c) for(uint32_t i=c*384;i<std::min(instances*3,(c+1)*384);++i) for(int a=0;a<3;++a)
            if(expected[i][a]<bounds[c*2][a]-.002f || expected[i][a]>bounds[c*2+1][a]+.002f || expected[i][a]<h->lo[a]-.002f || expected[i][a]>h->hi[a]+.002f) throw std::runtime_error("bounds mismatch");
        std::cout<<"PASS "<<instances<<" instances, "<<(run%2 && nq>1?"async":"same queue")<<"; positions, indices, chunk/header bounds\n";
    }
    vkDeviceWaitIdle(device);vkDestroyFence(device,fence,nullptr);vkDestroySemaphore(device,ready,nullptr);vkDestroyCommandPool(device,cp,nullptr);vkDestroyPipeline(device,pipeline,nullptr);vkDestroyShaderModule(device,shader,nullptr);vkDestroyPipelineLayout(device,pl,nullptr);vkDestroyDescriptorPool(device,pool,nullptr);vkDestroyDescriptorSetLayout(device,layout,nullptr);
    for(auto& b:buffers){vkUnmapMemory(device,b.m);vkDestroyBuffer(device,b.b,nullptr);vkFreeMemory(device,b.m,nullptr);}vkDestroyDevice(device,nullptr);vkDestroyInstance(instance,nullptr);return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
