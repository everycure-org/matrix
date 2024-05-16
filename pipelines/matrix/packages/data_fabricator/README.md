# Data Fabricator

The data fabricator utility was created to enable mocking a set of tables declaratively,
where join integrity between tables are easy to define and maintain. This enables
full integration tests of pipelines to be conducted and scaled without manually
crafting single data points for every single table.

Many libraries such as [faker](https://github.com/joke2k/faker),
[hypothesis](https://github.com/HypothesisWorks/hypothesis/tree/master/hypothesis-python),
or even the newer [GAN](https://github.com/sdv-dev/TGAN) based approaches
address the issue of mocking a **single** table realistically or rely on having a
dataset beforehand. `data_fabricator` works without the need for real data, only knowledge 
of such.

For more details please read [overview](src/data_fabricator/docs/00_overview.ipynb)