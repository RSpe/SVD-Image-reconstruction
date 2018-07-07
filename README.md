# SVD-reconstruction

1. Deconstruciton of black and white image (me2.pgm) via SVD showing the image in the form A = UDV^-T

2. Then illustrate the resulting approximation of A at certain points when we increase the value of p (where p is equal to the singular values/columns of U and V)

3. Outputing a table which tells us the compression levels, max error, mean error at each p value (shown above). This is giving us the statistics of our approximated A compared to our original A.

4. My thoughts on why there is negative compression at greater p values.

5. What I think is an optimal p value is when comparing the compression (size of the approximation), and the quality of the image compared the to the orginal image.

If you need an explanation of what SVD is: https://en.wikipedia.org/wiki/Singular_value_decomposition
