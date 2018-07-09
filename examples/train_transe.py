from openke import Dataset
from openke.models import TransE

# Read the dataset
base = Dataset("./benchmarks/fb15k.nt")

#   Set the knowledge embedding model class.
model = lambda: TransE(50, 1.0, base.shape, batchshape=(len(base) // 20, 2))

#   Train the model. It is saved in the process.
model, records = base.train(model, folds=20, epochs=20, batchkwargs={'negatives': (1, 0), 'bern': False, 'workers': 4},
                            eachepoch=print, prefix="./result")
print(records)

# test = Dataset("./benchmarks/FB15K/test2id.txt", E, R)
# print(test.meanrank(model, head=False, label=False))
