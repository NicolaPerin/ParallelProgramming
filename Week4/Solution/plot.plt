unset colorbox
set palette rgb 33,13,10
set size square
plot 'solution.dat' binary format='%double%double%double' using 1:2:3 with image
