This repository contains the data and analysis files for:

J. Denis, N. Villeneuve, P. Cacciagli, C. Mignon-Ravix, C. Lacoste, J. Lefranc, S. Napuri, L. Damaj, F. Villega, J-M. Pedespan, S. Moutton, C. Mignot, D. Doummar, L. Lion-François, S. Gataullina, O. Dulac, M Martin, S. Gueden, G. Lesca, S. Julia, C. Cances, H. Journel, C. Altuzarra, B. Ben Zeev, A. Afenjar,  M. Barth,  Laurent Villard, M. Milh (2019) [Clinical study of 19 patients with SCN8A-related epilepsy: two modes of onset regarding EEG and seizures] (accepted in Epilepsia) 

## Data

This repository contains the file SCN8A.patients.csv containing the data used for analysis concerning all patients from our cohort. 

The significations of the values used for each field are contained in the file data/codes_label.xlsx


## Analysis code

### Requirements

- [Python] code has been tested with v3.6 using anaconda distribution. 

### Rerunning analyses

You can re-run the analyses using the Python script (main_SCN8A.py).

To generate figures using the python script, execute the function main. First, you'll need to specify the path for the csv file, on the first line of the function main. You will also need to specify in which folder the figures should be saved, by setting the value of the variable 'path_results' at the beginning of the function main. 


## License

### Code

The analysis code is made available under the MIT license:

> Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
