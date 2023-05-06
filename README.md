# ProtoPool: Interpretable Image Classification with Differentiable Prototypes Assignment (Fork)

This is a fork of the [ProtoPool](https://github.com/gmum/ProtoPool) project created by D. Rymarczyk, Ł. Struski, M. Górszczak, K. Lewandowska, J. Tabor, and B. Zieliński. The original project is the implementation of the paper published at ECCV 2022.

The original code is based on the following repositories:

- https://github.com/cfchen-duke/ProtoPNet
- https://github.com/M-Nauta/ProtoTree
- https://github.com/gmum/ProtoPShare

## Purpose of this Fork

The purpose of this fork is to conduct some analysis on the ProtoPool model.

## How to Run

To run the code in this repository, follow these steps:

1. Create a new conda environment using the provided `environment.yml` file:
<pre>
```
conda env create -f environment.yml
```
</pre>

2. Activate the new conda environment:
<pre>
```
conda activate proto_pool
```
</pre>

3. Give execute permissions to the shell script:
<pre>
```
chmod +x run_cub_200_2011.sh
```
</pre>

4. Run the shell script:
<pre>
```
./run_cub_200_2011.sh
```
</pre>

## Original README

https://github.com/gmum/ProtoPool/blob/main/README.md

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/michalskit/ProtoPool/blob/main/license.txt) file for details.

## Acknowledgments

- Thanks to D. Rymarczyk, Ł. Struski, M. Górszczak, K. Lewandowska, J. Tabor, and B. Zieliński for creating the original project and their valuable work.



