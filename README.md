# Field Mapper (Field of The World)
## Project Summary
This repository shows the pipelines implemented for performing transductive learning leveraging the semantic information in the Field of the World Datasets transferred to selected areas in Africa, built on the Lacuna Funds Labels hosted by the Agricultural Impacts Research Group at Clark University for the Task of Crop Boundary Segmentation. Large scale crop boundary segmentation datasets are often scarce over the African region. For example, the Field of the World Crop Boundary Segmentation datasets, covers only two selected countries in the African region. In this project, we experiment with the large scale datasets and their ability to adequately transfer to new regions using datasets of different modalities. Hence, this repository implements boundary segmentation over selected areas in Africa leveraging pretrained models based of the Field of the World datasets.

## Methodology 
The approach taken in this project is implemented in two phases: 
1. Transferring the weights of the FTW Models (UNet Attention and DeepLab v3+) to the African Labels from Lacuna Fund, 
2. End to End Training using the Models from Agricultural Impact Research Group and the FTW datasets and testing different transfer learning scenarios for the African Labels from Lacuna Fund. 

## Datasets description
### Fields of the World
[Fields of the World](https://fieldsofthe.world)
### Lacuna Labels
[Lacuna Labels](https://github.com/agroimpacts/lacunalabels)

## How to run pipeline
1. How to run notebook in an attached docker file, 
2. Report in Quarto

## Contributors
1. Rufai Omowunmi Balogun
2. Tanmoy Chakraborty
3. Isaiah Taylor

## References
1. [A region-wide, multi-year set of crop field boundary labels for Africa](https://zenodo.org/records/11060871)
2. [Fields of the World (FTW): A Comprehensive Benchmark Datasets for Agricultural Field Boundaries](https://fieldsofthe.world)
