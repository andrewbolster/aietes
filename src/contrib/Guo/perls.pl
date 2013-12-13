system("perl perls-slip.pl 0 5");
system("perl perls-slip.pl 2 5");
system("perl perls-slip.pl 3 5");
system("perl perls-slip.pl 4 5");
system("perl perls-slip.pl 5 5");

system("perl grey-sum.pl 0");
system("perl grey-sum.pl 2");
system("perl grey-sum.pl 3");
system("perl grey-sum.pl 4");
system("perl grey-sum.pl 5");

system("perl grey-slip.pl 0");
system("perl grey-slip.pl 2");
system("perl grey-slip.pl 3");
system("perl grey-slip.pl 4");
system("perl grey-slip.pl 5");

system("perl total-slip.pl topo-relate-0-1 1");
system("perl total-slip-1225.pl topo-relate-0-1 1");
system("perl total-slip-slip.pl topo-relate-0-1 1");
system("perl total-slip-1225-slip.pl topo-relate-0-1 1");
exit(0);
