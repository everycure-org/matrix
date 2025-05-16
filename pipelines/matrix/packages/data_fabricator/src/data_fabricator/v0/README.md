# Data Fabricator

The data fabricator utility was created to enable mocking a set of tables declaratively,
where join integrity between tables are easy to define and maintain. This enables
full integration tests of pipelines to be conducted and scaled without manually
crafting single data points for every single table.

Many libraries such as [faker](https://github.com/joke2k/faker),
[hypothesis](https://github.com/HypothesisWorks/hypothesis/tree/master/hypothesis-python),
or even the newer [GAN](https://github.com/sdv-dev/TGAN) based approaches
address the issue of mocking a **single** table realistically. However, none natively
enables mocking a set of related tables.

For more details please read [overview](../docs/v0/00_overview.ipynb)