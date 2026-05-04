/* =============================================================================
 * MARL IoT Load Balancing — NS-3.43
 * Author: Prakriti
 *
 * TOPOLOGY:
 *   IoT Devices --[CSMA]--> Base Stations --[P2P]--> Gateway
 *
 * FIX: Using CSMA (per-BS LAN) instead of shared WiFi channel
 *      to avoid GlobalRouter network number confusion.
 * =============================================================================
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/csma-module.h"
#include "ns3/applications-module.h"
#include "ns3/mobility-module.h"
#include "ns3/flow-monitor-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("MarlIoTTopology");

// ============================================================
// PARAMETERS
// ============================================================
uint32_t NUM_IOT_DEVICES   = 10;
uint32_t NUM_BASE_STATIONS = 3;
double   SIM_TIME          = 30.0;
uint32_t PACKET_SIZE       = 512;
double   DATA_RATE_BPS     = 50000.0;
uint32_t LOAD_BALANCE_MODE = 1;    // 0=Static  1=RoundRobin
bool     ENABLE_FAILURE    = false;
uint32_t FAILED_BS_INDEX   = 1;
double   FAILURE_TIME      = 10.0;

// ============================================================
// Assign IoT device to Base Station
// ============================================================
uint32_t AssignBS(uint32_t i, uint32_t numBS, uint32_t mode)
{
    if (mode == 0) return 0;
    return i % numBS;
}

// ============================================================
// Simulate Base Station failure
// ============================================================
void FailBS(Ptr<Node> bs, uint32_t idx)
{
    NS_LOG_UNCOND("=== [FAULT] BS-" << idx
                  << " failed at t=" << Simulator::Now().GetSeconds() << "s ===");
    for (uint32_t i = 0; i < bs->GetNApplications(); i++)
        bs->GetApplication(i)->SetStopTime(Simulator::Now());
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char *argv[])
{
    CommandLine cmd(__FILE__);
    cmd.AddValue("numIoT",  "Number of IoT devices",   NUM_IOT_DEVICES);
    cmd.AddValue("numBS",   "Number of base stations", NUM_BASE_STATIONS);
    cmd.AddValue("simTime", "Simulation time (s)",     SIM_TIME);
    cmd.AddValue("lbMode",  "0=Static 1=RoundRobin",  LOAD_BALANCE_MODE);
    cmd.AddValue("failure", "Enable BS failure",       ENABLE_FAILURE);
    cmd.Parse(argc, argv);

    NS_LOG_UNCOND("============================================");
    NS_LOG_UNCOND(" MARL IoT Simulation Starting");
    NS_LOG_UNCOND(" IoT Devices  : " << NUM_IOT_DEVICES);
    NS_LOG_UNCOND(" Base Stations: " << NUM_BASE_STATIONS);
    NS_LOG_UNCOND(" Duration     : " << SIM_TIME << "s");
    NS_LOG_UNCOND(" LB Mode      : " << (LOAD_BALANCE_MODE==0?"Static":"RoundRobin"));
    NS_LOG_UNCOND(" Failure Test : " << (ENABLE_FAILURE?"YES":"NO"));
    NS_LOG_UNCOND("============================================");

    // ---- 1. Create Nodes ----
    NodeContainer iotNodes, bsNodes, gwNode;
    iotNodes.Create(NUM_IOT_DEVICES);
    bsNodes.Create(NUM_BASE_STATIONS);
    gwNode.Create(1);

    // ---- 2. Internet Stack ----
    InternetStackHelper internet;
    internet.Install(iotNodes);
    internet.Install(bsNodes);
    internet.Install(gwNode);

    // ---- 3. Assign IoT → BS groups ----
    std::vector<NodeContainer> groups(NUM_BASE_STATIONS);
    for (uint32_t i = 0; i < NUM_IOT_DEVICES; i++)
    {
        uint32_t b = AssignBS(i, NUM_BASE_STATIONS, LOAD_BALANCE_MODE);
        groups[b].Add(iotNodes.Get(i));
        NS_LOG_UNCOND("[ASSIGN] IoT-" << i << " -> BS-" << b);
    }

    // ---- 4. CSMA LAN per BS (IoT <-> BS) ----
    // Each BS gets its own LAN — no shared channel confusion
    CsmaHelper csma;
    csma.SetChannelAttribute("DataRate", StringValue("10Mbps"));
    csma.SetChannelAttribute("Delay",    TimeValue(MilliSeconds(1)));

    Ipv4AddressHelper ipv4;
    std::vector<Ipv4InterfaceContainer> bsLanIfaces(NUM_BASE_STATIONS);

    for (uint32_t b = 0; b < NUM_BASE_STATIONS; b++)
    {
        // Build LAN: BS node + its IoT nodes
        NodeContainer lan;
        lan.Add(bsNodes.Get(b));
        lan.Add(groups[b]);

        NetDeviceContainer lanDevs = csma.Install(lan);

        // Unique subnet per BS: 10.1.1.0, 10.1.2.0, 10.1.3.0 ...
        std::string subnet = "10.1." + std::to_string(b+1) + ".0";
        ipv4.SetBase(subnet.c_str(), "255.255.255.0");
        bsLanIfaces[b] = ipv4.Assign(lanDevs);

        NS_LOG_UNCOND("[LAN] BS-" << b
                      << " subnet=" << subnet
                      << " nodes=" << groups[b].GetN());
    }

    // ---- 5. P2P Backbone: BS <-> Gateway ----
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("100Mbps"));
    p2p.SetChannelAttribute("Delay",   StringValue("2ms"));

    std::vector<Ipv4InterfaceContainer> gwIfaces(NUM_BASE_STATIONS);
    for (uint32_t b = 0; b < NUM_BASE_STATIONS; b++)
    {
        NetDeviceContainer lnk = p2p.Install(bsNodes.Get(b), gwNode.Get(0));
        // Unique backbone subnet: 10.2.1.0, 10.2.2.0, 10.2.3.0 ...
        std::string subnet = "10.2." + std::to_string(b+1) + ".0";
        ipv4.SetBase(subnet.c_str(), "255.255.255.0");
        gwIfaces[b] = ipv4.Assign(lnk);
        NS_LOG_UNCOND("[P2P] BS-" << b << " <-> GW subnet=" << subnet);
    }

    // ---- 6. Mobility (positions for reference) ----
    MobilityHelper mob;
    mob.SetMobilityModel("ns3::ConstantPositionMobilityModel");

    Ptr<ListPositionAllocator> bsPos = CreateObject<ListPositionAllocator>();
    for (uint32_t b = 0; b < NUM_BASE_STATIONS; b++)
        bsPos->Add(Vector(b * 100.0, 0.0, 0.0));
    mob.SetPositionAllocator(bsPos);
    mob.Install(bsNodes);

    mob.SetPositionAllocator(
        "ns3::RandomRectanglePositionAllocator",
        "X", StringValue("ns3::UniformRandomVariable[Min=0|Max=300]"),
        "Y", StringValue("ns3::UniformRandomVariable[Min=-30|Max=30]"));
    mob.Install(iotNodes);

    Ptr<ListPositionAllocator> gwPos = CreateObject<ListPositionAllocator>();
    gwPos->Add(Vector(100.0, 150.0, 0.0));
    mob.SetPositionAllocator(gwPos);
    mob.Install(gwNode);

    // ---- 7. Traffic: IoT → Gateway ----
    // Gateway IP = second interface on first BS-GW link
    Ipv4Address gwAddr = gwIfaces[0].GetAddress(1);
    NS_LOG_UNCOND("[TRAFFIC] Gateway IP: " << gwAddr);

    // UDP Server on Gateway
    UdpServerHelper server(9);
    ApplicationContainer srvApp = server.Install(gwNode.Get(0));
    srvApp.Start(Seconds(0.0));
    srvApp.Stop(Seconds(SIM_TIME));

    // UDP Clients on each IoT device
    double interval = (PACKET_SIZE * 8.0) / DATA_RATE_BPS;
    for (uint32_t i = 0; i < NUM_IOT_DEVICES; i++)
    {
        UdpClientHelper client(gwAddr, 9);
        client.SetAttribute("MaxPackets", UintegerValue(100000));
        client.SetAttribute("Interval",   TimeValue(Seconds(interval)));
        client.SetAttribute("PacketSize", UintegerValue(PACKET_SIZE));
        ApplicationContainer app = client.Install(iotNodes.Get(i));
        app.Start(Seconds(1.0 + i * 0.05));
        app.Stop(Seconds(SIM_TIME));
    }
    NS_LOG_UNCOND("[TRAFFIC] " << NUM_IOT_DEVICES
                  << " IoT clients sending every " << interval << "s");

    // ---- 8. Fault Tolerance ----
    if (ENABLE_FAILURE && FAILED_BS_INDEX < NUM_BASE_STATIONS)
    {
        NS_LOG_UNCOND("[FAULT] Scheduling BS-" << FAILED_BS_INDEX
                      << " failure at t=" << FAILURE_TIME << "s");
        Simulator::Schedule(Seconds(FAILURE_TIME),
                            &FailBS,
                            bsNodes.Get(FAILED_BS_INDEX),
                            FAILED_BS_INDEX);
    }

    // ---- 9. Flow Monitor + Routing ----
    FlowMonitorHelper flowmon;
    Ptr<FlowMonitor> monitor = flowmon.InstallAll();
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();

    // ---- 10. Run ----
    NS_LOG_UNCOND("--------------------------------------------");
    NS_LOG_UNCOND("[SIM] Running " << SIM_TIME << "s simulation...");
    NS_LOG_UNCOND("--------------------------------------------");
    Simulator::Stop(Seconds(SIM_TIME));
    Simulator::Run();

    // ---- 11. Collect Results ----
    monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier =
        DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());
    FlowMonitor::FlowStatsContainer stats = monitor->GetFlowStats();

    double   totalTput=0, totalLat=0;
    uint64_t totalTx=0, totalRx=0;
    uint32_t fc=0;
    std::vector<double> bsLoad(NUM_BASE_STATIONS, 0.0);

    for (auto& flow : stats)
    {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(flow.first);
        fc++;
        double dur  = SIM_TIME - 1.0;
        double tput = flow.second.rxPackets > 0 ?
                      (flow.second.rxBytes*8.0)/(dur*1e6) : 0;
        double lat  = flow.second.rxPackets > 0 ?
                      (flow.second.delaySum.GetSeconds()/
                       flow.second.rxPackets)*1000.0 : 0;
        double loss = flow.second.txPackets > 0 ?
                      100.0*(flow.second.txPackets-flow.second.rxPackets)/
                      flow.second.txPackets : 0;

        totalTput += tput;
        totalLat  += lat;
        totalTx   += flow.second.txPackets;
        totalRx   += flow.second.rxPackets;
        bsLoad[fc % NUM_BASE_STATIONS] += tput;

        NS_LOG_UNCOND("Flow-" << fc
            << " " << t.sourceAddress << "->" << t.destinationAddress
            << " Tput=" << tput << "Mbps"
            << " Lat="  << lat  << "ms"
            << " Loss=" << loss << "%");
    }

    // ---- Summary for Paper ----
    double avgLat  = fc > 0 ? totalLat/fc : 0;
    double totLoss = 100.0*(totalTx-totalRx)/std::max(totalTx,(uint64_t)1);
    double sumX=0, sumX2=0;
    for (uint32_t b=0; b<NUM_BASE_STATIONS; b++)
    {
        sumX  += bsLoad[b];
        sumX2 += bsLoad[b] * bsLoad[b];
    }
    double jfi = (sumX*sumX) / (NUM_BASE_STATIONS*sumX2 + 1e-9);

    NS_LOG_UNCOND("============================================");
    NS_LOG_UNCOND("         PAPER METRICS SUMMARY");
    NS_LOG_UNCOND("============================================");
    NS_LOG_UNCOND("Flows                  : " << fc);
    NS_LOG_UNCOND("Total Throughput       : " << totalTput << " Mbps");
    NS_LOG_UNCOND("Avg Latency            : " << avgLat    << " ms");
    NS_LOG_UNCOND("Packet Loss            : " << totLoss   << " %");
    NS_LOG_UNCOND("Tx Packets             : " << totalTx);
    NS_LOG_UNCOND("Rx Packets             : " << totalRx);
    for (uint32_t b=0; b<NUM_BASE_STATIONS; b++)
        NS_LOG_UNCOND("BS-" << b << " Load              : "
                      << bsLoad[b] << " Mbps");
    NS_LOG_UNCOND("Jain Fairness Index    : " << jfi
                  << "  (1.0 = perfect)");
    NS_LOG_UNCOND("============================================");

    // Save XML for Python analysis
    monitor->SerializeToXmlFile("marl-iot-results.xml", true, true);
    NS_LOG_UNCOND("[OUTPUT] marl-iot-results.xml saved!");
    NS_LOG_UNCOND("[NEXT]   Run Python scripts to analyze & plot");

    Simulator::Destroy();
    return 0;
}