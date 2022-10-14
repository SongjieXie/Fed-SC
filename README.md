# Fed-SC: One-shot Federated Subspace Clustering

![avatar](./imgs/log.png)

Fed-SC is an one-shot federated scheme for subspace clustering, which 

* **Communication efficiency:** Fed-SC just needs one round communication, i.e., each device upload the generated samples to the central server and the server deliver the cluster assignments for local updating to obtain the final clustering.

* **Scalability:** The sequential and parallel running times of Fed-SC achieve reductions from , respectively.

* **Robustness:** Owing to the robustness of SSC and TSC implemented at the central server together with our generically designed cluster encoding and sampling procedures, Fed-SC can exhibit great robustness against communication noise and attacks.

* **Theoretical guarantees:** The final clustering is theoretically guaranteed under broad conditions on data distribution and subspace affinity.

## Installation

1. git clone ....  

## Dependencies

This project is compatible with the packages:

* pytorch 1.8.0

* torchvision 0.90a0

* numpy 1.23.1

The project can be run on the syn and real-world datasets such as ...

```python
root = r'./data/'
```

## Examples
