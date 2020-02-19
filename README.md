Generates structures for Science Birds (Angry Birds) from sketches of rectilinear polygons.

https://github.com/lucasnfe/Science-Birds

To generate structure from sketch:

python3 generate_sketch.py "sketch_name.png" "scale_calculation_option" "corner_splitting"

-   scale_calculation_option:       0=Max  1=Min  2=MidRange  3=Mean  4=Median
-   corner_splitting:      0=only rectangles within sketch  1=sketch contains non-rectangular polygons

e.g. 
python3 generate_sketch.py sketch.png 3 0

NOTE:
-   It may not be possible to generate a structure from every sketch.
-   Generate and compare the generated structures for each scale_calculation_option to get the best results.
-   If not all corners of the sketched polygons are being detected correctly, try changing the "corner_detection_quality_threshold" value

A good way to automatically create many structure sketches is using the random rectilinear polygon generator found here:
https://github.com/stepmat/rectilinear_polygon_generator


Input sketch | Generated structure
:-------------------------:|:-------------------------:
![](/other/do_not_use_for_generation/example1a.jpg) | ![](/other/do_not_use_for_generation/example1b.png)
![](/other/do_not_use_for_generation/example2a.png) | ![](/other/do_not_use_for_generation/example2b.png)
![](/other/do_not_use_for_generation/example3a.png) | ![](/other/do_not_use_for_generation/example3b.png)
![](/other/do_not_use_for_generation/example4a.jpg) | ![](/other/do_not_use_for_generation/example4b.png)
