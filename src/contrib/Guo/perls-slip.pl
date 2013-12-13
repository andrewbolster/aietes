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

perl grey-sum.pl
# every parameter
# every slip(seconds)
$node=$ARGV[0];
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
    $dr[$i]=$x[4];
    $i=$i+1;}
}
close FILE;

$isize=$i;
$i=0;
$j=0;
$k=0;
$size=0;
@a=[];

for($i=0;$i<$isize;$i++){
  $gfile="g".$udp[$i];
  open (DATA,"<$gfile")
  || die "Can't open $gfile $!";
  while (<DATA>) {
    @x = split(' ');
    $size=@x;
    if($k==0){
      $k=$k+1;}
    else{
      for($j=0;$j<$size;$j++){
        $a[$i][$k][$j]=$x[$j];}
      $k=$k+1;}
  }
  $knum=$k;
  $k=0;
  close DATA;
}

$jnum=$size;
for($i=0;$i<$isize;$i++){
  for($j=0;$j<$jnum;$j++){
    $a[$i][$knum][$j]=$dr[$i];
  }
}

$max=0;
$min=20000000;
@b=[];
@g=[];

for($j=0;$j<$jnum;$j++){
  for($k=1;$k<=$knum;$k++){
    for($i=0;$i<$isize;$i++){
      if($max<$a[$i][$k][$j]){
        $max=$a[$i][$k][$j];}
      if($min>$a[$i][$k][$j]){
        $min=$a[$i][$k][$j];}
    }
    for($i=0;$i<$isize;$i++){
      if ($max==$min){
        $b[$i][$k][$j]=1;
        $g[$i][$k][$j]=1;}
      else {
        $b[$i][$k][$j]=0.75*($max-$min)/(abs($a[$i][$k][$j]-$max)+0.5*($max-$min))-0.5;
        $g[$i][$k][$j]=0.75*($max-$min)/(abs($a[$i][$k][$j]-$min)+0.5*($max-$min))-0.5;
      }
    }
    $max=0;
    $min=20000000;
  }
}

$good=0;
$bad=0;
$value=0;
@ag=[];

for ($i=0;$i<$isize;$i++){
  for($j=0;$j<$jnum;$j++){
    $good=0;
    $bad=0;
    $value=0;
    for($k=1;$k<=$knum;$k++){
      if($k!=4){
        $good=$good+0.2*$g[$i][$k][$j];
        $bad=$bad+0.2*$b[$i][$k][$j];
      }
      if($k==4){
        $good=$good+0.2*$b[$i][$k][$j];
        $bad=$bad+0.2*$g[$i][$k][$j];
      }
    }
    $value=1/(1+$bad*$bad/($good*$good));
    $ag[$i][$j]=$value;
  }
}
open(RFILE,">v$ARGV[0]");
print RFILE "$ARGV[0] ";
for($i=0;$i<$isize;$i++){
  print RFILE "$noder[$i] ";
}
print RFILE "\n";
for($j=0;$j<$jnum;$j++){
  print RFILE "$j ";
  for($i=0;$i<$isize;$i++){
    print RFILE "$ag[$i][$j] ";
  }
  print RFILE "\n";
}
close RFILE;
exit(0);
