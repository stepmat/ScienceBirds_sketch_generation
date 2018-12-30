Generates structures for Science Birds (Angry Birds) from sketches of rectilinear polygons.

https://github.com/lucasnfe/Science-Birds

To generate structure from sketch:

python3 generate_sketch.py "sketch_name.png" "scale_calculation_option" "only_rectangular_polygons"

scale_calculation_option:       0=Max  1=Min  2=MidRange  3=Mean  4=Median
only_rectangular_polygons:      1=only rectangles within sketch  0=sketch contains non-rectangular polygons

e.g. 
python3 generate_sketch.py sketch.png 3 0

NOTE:
-   It may not be possible to generate a structure from every sketch.
-   Generate and compare the generated structures for each scale_calculation_option to get the best results.
-   If not all corners of the sketched polygons are being detected correctly, try changing the "corner_detection_quality_threshold" value

A good way to automatically create many structure sketches is using the random rectilinear polygon generator found here:
https://github.com/stepmat/rectilinear_polygon_generator

Input sketch             |  Generated structure
:-------------------------:|:-------------------------:
![](/generation_examples/1a.png) | ![](/generation_examples/1b.jpg)
![](/generation_examples/2a.jpg) | ![](/generation_examples/2b.jpg)
![](/generation_examples/3a.jpg) | ![](/generation_examples/3b.jpg)
![](/generation_examples/4a.png) | ![](/generation_examples/4b.jpg)