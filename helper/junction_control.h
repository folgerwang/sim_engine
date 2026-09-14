#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>
#include <utility>

namespace engine::helper {
// Collect arrivals before movement, then assign stable tickets. Distance and
// vehicle identity break same-frame ties without favoring vector iteration.
template<class Vehicles>
void assignJunctionTickets(Vehicles& cars, std::vector<std::pair<float,int>>& arrivals,
                           uint64_t& next) {
    std::sort(arrivals.begin(),arrivals.end(),[&](const auto& a,const auto& b){
        if(a.first!=b.first) return a.first<b.first;
        if(cars[a.second].seed!=cars[b.second].seed)
            return cars[a.second].seed<cars[b.second].seed;
        return a.second<b.second;
    });
    for(const auto& arrival:arrivals) cars[arrival.second].arrival_ticket=next++;
}

template<class Vehicles>
int junctionQueueHead(const Vehicles& cars, int node) {
    int head=-1;
    for(int i=0;i<int(cars.size());++i) {
        const auto& c=cars[i];
        if(c.parked || c.dormant || c.queue_node!=node || !c.arrival_ticket) continue;
        if(head<0 || c.arrival_ticket<cars[head].arrival_ticket) head=i;
    }
    return head;
}

template<class Vehicles>
bool junctionAdmission(const Vehicles& cars, int node, int vehicle, int holder,
                       bool stopped, bool exit_clear) {
    if(holder==vehicle) return true; // let an admitted car finish crossing
    const bool occupied=holder>=0 && holder<int(cars.size()) &&
        !cars[holder].parked && !cars[holder].dormant && cars[holder].claim==node;
    return !occupied && stopped && exit_clear && junctionQueueHead(cars,node)==vehicle;
}
} // namespace engine::helper
