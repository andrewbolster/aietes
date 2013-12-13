
# Define options defining some variables
set val(chan) Channel/WirelessChannel ;# channel type
set val(prop) Propagation/FreeSpace ;# radio-propagation model
set val(netif) Phy/WirelessPhy ;# network interface type
Phy/WirelessPhy set CPThresh_ 10.0
Phy/WirelessPhy set CSThresh_ 4.4619e-10 ;#250m, freespace model
Phy/WirelessPhy set RXThresh_ 4.4619e-10 ;#250m
Phy/WirelessPhy set bandwidth_ 1Mb ;
Phy/WirelessPhy set Pt_ 0.2818 ; #281.8mW
Phy/WirelessPhy set freq_ 2.4e+9 ;
Phy/WirelessPhy set L_ 1.0
set val(mac) Mac/802_11 ;# MAC type
#Mac/802_11 set dataRate_ 11.0e6 ;#data rate
#Mac/802_11 set dataRate_ 54.0e6 ;#data rate

set val(ifq) Queue/DropTail/PriQueue ;# interface queue type
set val(ll) LL ;# link layer type
set val(ant) Antenna/OmniAntenna ;# antenna model

Antenna/OmniAntenna set X_ 0
Antenna/OmniAntenna set Y_ 0
Antenna/OmniAntenna set Z_ 0.25
Antenna/OmniAntenna set Gt_ 1
Antenna/OmniAntenna set Gr_ 1

set val(x) 700 ;# X dimension of topology
set val(y) 1000 ;# Y dimension of topology
set val(cp) "" ;# node movement model file
set val(sc) "" ;# traffic model file
set val(ifqlen) 50 ;# max packet in ifq
set val(nn) 6 ;# number of nodes, one mobile node1
set val(seed) 0.0
set val(stop) 305.0 ;#
set val(tr) nam-out.tr ;# trace file name

set val(rp) DSDV ;# routing protocol
set AgentTrace ON
set RouterTrace ON
set MacTrace OFF

#set global variables
set ns [new Simulator]
#Mac/802_11 set RTSThreshold_ 0

$ns color 1 green
$ns color 2 red

#open trace file
$ns use-newtrace
set namfd [open out.nam w]
$ns namtrace-all-wireless $namfd $val(x) $val(y)
set tracefd [open $val(tr) w]
$ns trace-all $tracefd

#set up a topology
set topo [new Topography]
$topo load_flatgrid $val(x) $val(y)
set chan [new $val(chan)]

#set a god
set god [create-god $val(nn)]
$ns node-config -adhocRouting $val(rp) \
-llType $val(ll) \
-macType $val(mac) \
-ifqType $val(ifq) \
-ifqLen $val(ifqlen) \
-antType $val(ant) \
-propType $val(prop) \
-phyType $val(netif) \
-channel $chan \
-topoInstance $topo \
-agentTrace ON \
-routerTrace ON \
-macTrace OFF \
-movementTrace OFF

for {set i 0} {$i < $val(nn) } {incr i} {
set node_($i) [$ns node]
$node_($i) random-motion 0

;# disable random motion

}

# Provide initial (X,Y, for now Z=0) co-ordinates for mobile nodes, here all static
###set one node the command "$ ns node" only one time
#set node_(0) [$ns node]
#$node_(0) random-motion 0

# node0 (400.0, 500.0)
$node_(0) set X_ 400.0
$node_(0) set Y_ 500.0
$node_(0) set Z_ 0.0

# node1 (500.0, 500.0)
$node_(1) set X_ 500.0
$node_(1) set Y_ 500.0
$node_(1) set Z_ 0.0

# node2 (450.0, 450.0)
$node_(2) set X_ 450.0
$node_(2) set Y_ 450.0
$node_(2) set Z_ 0.0

# node3 (450.0, 550.0)
$node_(3) set X_ 450.0
$node_(3) set Y_ 550.0
$node_(3) set Z_ 0.0

# node4
$node_(4) set X_ 600.0
$node_(4) set Y_ 470.0
$node_(4) set Z_ 0.0

# node5
$node_(5) set X_ 600.0
$node_(5) set Y_ 530.0
$node_(5) set Z_ 0.0

# Load the god object with shortest hop information

# 0 and 1 hop=1
$god set-dist 0 1 1
# 0 and 2 hop=1
$god set-dist 0 2 1
# 0 and 3 hop=1
$god set-dist 0 3 1
# 1 and 2 hop=1
$god set-dist 1 2 1
# 1 and 3 hop=1
$god set-dist 1 3 1
# 1 and 4 hop=1
$god set-dist 1 4 1
# 1 and 5 hop=1
$god set-dist 1 5 1
# 2 and 4 hop=1
$god set-dist 2 4 1
# 3 and 5 hop=1
$god set-dist 3 5 1
# 4 and 5 hop=1
$god set-dist 4 5 1

#===================================
#

Agents Definition

#===================================
#Setup a CBR Application over UDP connection
set node_mac_(0) [$node_(0) set mac_(0)]
$node_mac_(0) set dataRate_ 1.0e6
$node_mac_(0) set basicRate_ 1.0e6
set node_mac_(1) [$node_(1) set mac_(0)]
$node_mac_(1) set dataRate_ 1.0e6
$node_mac_(1) set basicRate_ 1.0e6
set node_mac_(2) [$node_(2) set mac_(0)]
$node_mac_(2) set dataRate_ 1.0e6
$node_mac_(2) set basicRate_ 1.0e6
set node_mac_(3) [$node_(3) set mac_(0)]
$node_mac_(3) set dataRate_ 1.0e6
$node_mac_(3) set basicRate_ 1.0e6
set node_mac_(4) [$node_(4) set mac_(0)]
$node_mac_(4) set dataRate_ 1.0e6
$node_mac_(4) set basicRate_ 1.0e6
set node_mac_(5) [$node_(5) set mac_(0)]
$node_mac_(5) set dataRate_ 1.0e6
$node_mac_(5) set basicRate_ 1.0e6

# 0 connecting to 1 at time 10
set udp_(01) [new Agent/mUDP]
$udp_(01) set_filename sd_udp_01
$udp_(01) set fid_ 1
$ns attach-agent $node_(0) $udp_(01)
set null_(01) [new Agent/mUdpSink]
$null_(01) set_filename rd_udp_10
$ns attach-agent $node_(1) $null_(01)
set cbr_(01) [new Application/Traffic/CBR]
$cbr_(01) set packetSize_ 512
$cbr_(01) set random_ 1
$cbr_(01) set maxpkts_ 10000
$cbr_(01) set rate_ 1.7Mb
$cbr_(01) attach-agent $udp_(01)
$ns connect $udp_(01) $null_(01)

$ns at 10 "$cbr_(01) start"
$ns at 11 "$cbr_(01) stop"
$ns at 15 "$cbr_(01) start"
$ns at 16 "$cbr_(01) stop"
$ns at 20 "$cbr_(01) start"
$ns at 21 "$cbr_(01) stop"
$ns at 25 "$cbr_(01) start"
$ns at 26 "$cbr_(01) stop"
$ns at 30 "$cbr_(01) start"
$ns at 31 "$cbr_(01) stop"
$ns at 35 "$cbr_(01) start"
$ns at 36 "$cbr_(01) stop"
$ns at 40 "$cbr_(01) start"
$ns at 41 "$cbr_(01) stop"
$ns at 45 "$cbr_(01) start"
$ns at 46 "$cbr_(01) stop"
$ns at 50 "$cbr_(01) start"
$ns at 51 "$cbr_(01) stop"
$ns at 55 "$cbr_(01) start"
$ns at 56 "$cbr_(01) stop"

$ns at 10 "$node_mac_(0) set dataRate_ 2e6"
$ns at 15 "$node_mac_(0) set dataRate_ 2e6"
$ns at 20 "$node_mac_(0) set dataRate_ 2e6"
$ns at 25 "$node_mac_(0) set dataRate_ 2e6"
$ns at 30 "$node_mac_(0) set dataRate_ 2e6"
$ns at 35 "$node_mac_(0) set dataRate_ 2e6"
$ns at 40 "$node_mac_(0) set dataRate_ 2e6"
$ns at 45 "$node_mac_(0) set dataRate_ 2e6"
$ns at 50 "$node_mac_(0) set dataRate_ 2e6"
$ns at 55 "$node_mac_(0) set dataRate_ 2e6"

set filepr1 [open outpr0-1.tr w]
#to calculate pr
set pt 0.2818
set l 1.0
set lambda 0.125
set pi 3.1415926
set gt 1.0
set gr 1.0
set ht 0.25
set hr 0.25

proc record1 {} {
global filepr1 pt l lambda pi gt gr ht hr node_

set ns [Simulator instance]
set time 1.0;# record every 1.0 second
set m_x [$node_(0) set X_]
set m_y [$node_(0) set Y_]
set n_x [$node_(1) set X_]
set n_y [$node_(1) set Y_]
set d [expr (sqrt(pow(($m_x - $n_x),2)+ pow(($m_y-$n_y),2)))]
set m [expr ($lambda /(4 * $pi * $d))]
set pr [expr ($pt * $gr * $gt * $m * $m / $l)]
set now [$ns now]
puts $filepr1 "[expr $now + $time]\t0\t1\t$d\t$pr"
$ns at [expr $now + $time] "record1"}
$ns at 10 "record1"

# node_i connecting to node_j at time t
……
#Define node initial position in nam, only for nam
# set initial position in nam
for {set i 0} {$i < $val(nn)} {incr i} {
# The function must be called after mobility model is defined.
$ns initial_node_pos $node_($i) 60
}

# Tell nodes when the simulation ends
# set end time
for {set i 0} {$i < $val(nn) } {incr i} {
$ns at $val(stop)

"$node_($i) reset";

}
$ns at $val(stop) "$ns nam-end-wireless $val(stop)"
$ns at $val(stop) "stop"
$ns at $val(stop) "puts \"NS EXITING...\"; $ns halt"
proc stop {} {
global ns tracefd namfd
$ns flush-trace
close $tracefd
close $namfd
exec nam out.nam &
exit 0 }

puts "Starting Simulation..."
$ns run

perls-slip.pl
$node=$ARGV[0];
$slip=$ARGV[1];
@noder=();
@udp=();
@start=();
$i=0;
$topo="topo-static";

open (FILE,"<$topo")
|| die "Can't open $topo $!";
while (<FILE>) {
@x = split(' ');
if($x[1]==$node){
$noder[$i]=$x[2];
$udp[$i]=$x[0];
$start[$i]=$x[3];
$i=$i+1;}
}
close FILE;

$isize=$i;
for($i=0;$i<$isize;$i++){
$cbr=$udp[$i];
$re_node=$noder[$i];
$starttime=$start[$i];
$endtime=$start[$i]+50;
$end=$start[$i]+50;
$sd="sd_udp_".$cbr;
$rd="rd_udp_".$re_node.$node;
$outpr="outpr".$node."-".$re_node.".tr";
$grey="grey".$cbr;
$g="g".$cbr;
$nam="nam-out.tr";
print STDOUT "\nfor node $node\n";
print STDOUT "\nneighbour node $re_node\n";
system("perl rs-slip.pl $rd $sd $starttime $end $slip $grey $g");
system("perl ss-slip.pl $outpr $starttime $end $slip $grey $g");
system("perl delay-slip.pl $rd $starttime $end $slip $grey $g");
system("perl measure-throughput-slip.pl $nam $starttime $endtime $slip $re_node $grey $g");
}
exit(0);
