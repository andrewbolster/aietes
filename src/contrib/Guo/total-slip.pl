# total value
$infile=$ARGV[0];
$node=$ARGV[1];
@relate=();
@relate_node=();
$k=0;

open (RDATA,"<$infile")
|| die "Can't open $infile $!";
while (<RDATA>) {
  @x = split(' ');
  $relate[$k]=$x[0];
  $relate_node[$k]=$x[1];
  $k=$k+1;
}
$knum=$k;
close RDATA;

@t=();
for($k=0;$k<$knum;$k++){
  $valuefile="v".$relate_node[$k];
  open (DATA,"<$valuefile")
  || die "Can't open $valuefile $!";
  $true=0;
  $size=0;
  $j=0;
  while (<DATA>) {
    @x = split(' ');
    $size=@x;
    if($true==0){
      for($i=0;$i<$size;$i++){
        if($x[$i]==$node){
          $inode=$i;
        }
      }
      $true=1;
      $j=$j+1;
    }
    else {
      $a[$k][$j]=$x[$inode];
# record time
      $t[$j]=$x[0];
      $j=$j+1;
    }
  }
  $jnum=$j;
  close DATA;
}

$d=0;
$r=0;
$id=0;
for ($k=0;$k<$knum;$k++){
  if($relate[$k]==0){
    $d=$d+1;
  }
  if($relate[$k]==1){
    $r=$r+1;
  }
  if($relate[$k]==2){
    $id=$id+1;
  }
}

@w=(0.5*$d,0.5*2*$r/((2*$r+$id)*$r),0.5*$id/((2*$r+$id)*$id));
print STDOUT "\nweights: $w[0] $w[1] $w[2] \n";
print STDOUT "total value: \n";
$total=0;
$max=0;

for($j=1;$j<$jnum;$j++){
  $total=0;
  for($k=0;$k<$knum;$k++){
    $f[0]=1-$a[$k][$j];
    $f[2]=$a[$k][$j];
    if($a[$k][$j]<=0.5){
      $f[1]=2*$a[$k][$j];
    }
    if($a[$k][$j]>0.5){
      $f[1]=2-2*$a[$k][$j];
    }
    $max=0;
    for($i=0;$i<3;$i++){
      if($max<$f[$i]){
        $max=$f[$i];
      }
    }
    if($relate[$k]==0){
      $total=$total+$a[$k][$j]*$max*$w[0];
      print STDOUT "$a[$k][$j]

      \n";

    }
    if($relate[$k]==1){
      $total=$total+$a[$k][$j]*$max*$w[1];
      print STDOUT "$a[$k][$j]

      \n";

    }
    if($relate[$k]==2){
      $total=$total+$a[$k][$j]*$max*$w[2];
      print STDOUT "$a[$k][$j]

      \n";

    }
  }
  print STDOUT "5nodes \n";
  print STDOUT "$total \n";
}
exit(0);
