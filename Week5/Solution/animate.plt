set term png
set output "diffusivity1.png"
minT=0
maxT=0.9
set cbrange [minT:maxT]
plot 'diffusivity1.dat' matrix with image

set term png
set output "diffusivity2.png"
minT=0
maxT=0.9
set cbrange [minT:maxT]
plot 'diffusivity2.dat' matrix with image

set term png
set output "diffusivity3.png"
minT=0
maxT=0.9
set cbrange [minT:maxT]
plot 'diffusivity3.dat' matrix with image

set term png
set output "concentration_init.png"
minT=0
maxT=0.9
set cbrange [minT:maxT]
plot 'concentration_init.dat' matrix with image

set term gif animate delay 25
set output "animate.gif"
frames = 20
minT=0
maxT=0.1
set cbrange [minT:maxT]
plot 'concentration_init.dat' matrix with image

do for [i=1:frames] {
  index_val = 1 + (i - 1) * 3
  file_name = sprintf("concentration_%d.dat", index_val)
  plot file_name matrix with image
}
