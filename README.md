# Learned Low Rank
 Learned Low Rank Prior



* Using only the low rank Casorati matrix property and do not using any CNN Net, just an unfolding version of the algorithm which using ADMM to solve the following optimization problem: [equation](https://www.latexlive.com/#JTIwJTIwJCQlMEElMjAlMjAlNUNtaW5feCUyMCU1Q2ZyYWMlMjAxMiUyMCU1Q1ZlcnQlMjBTRngtYiU1Q1ZlcnRfMiU1RTIrJTVDbGFtYmRhJTIwJTVDVmVydCUyMHglMjAlNUNWZXJ0XyolMEElMjAlMjAkJA==)

* referred from [Keziwen/SLR-Net: Code for our work: "Learned Low-rank Priors in Dynamic MR Imaging" (github.com)](https://github.com/Keziwen/SLR-Net)

* the paper of SLR-Net is

```
Ke, Z., Huang, W., Cui, Z. X., Cheng, J., Jia, S., Wang, H., ... & Liang, D. (2021). 
Learned Low-rank Priors in Dynamic MR Imaging. 
IEEE Transactions on Medical Imaging, DOI: 10.1109/TMI.2021.3096218.
```



#### The Files of this project

* `main.py` is the training code, and `test.py` is the testing code
* `model.py` is the unfolding network code
* `dataset_tfrecord.py` is the code for loading data from `*.tfrecord` files
* `WriteTFRecord.py` & `WriteTFRecord_singleCoil.py` are the codes for making the `*.tfrecord` files from [OCMR](https://ocmr.info/) dataset BY MYSELF

#### NOTE

* The derivation of the algorithm which using ADMM to solve the low-rank optimization problem can be find in `derivation.md` 
