# Using Postgres DB for storing flattened Model Output Data and Building Data access API using python based FastAPI.

## Attendees 
- **Everycure** :  Pascal B , Laurens V
- **Thoughtworks**: Manisha S , Masiur R , Akshay S
- **Status**: Accepted
- **Date**: 2024-10-28
- **Decision ID**: ADR-001

## Highlevel Data flow and Design

- On ML model output update , a trigger based data aggregation and flattening would initiate and dump drug:disease:treatscore data into a DB of choice. This job will do necessary lookups to gather disease and drug related other columns and create a new version of table in Database.
- At the same time , two separate tables containing unique list of drugs and diseases would also be refreshed. Assumption is the record count for these tables would be in 5k-20k range to begin with.
- Data Model in DB would be designed around Drug and Disease names as important columns for query filtering and search, hence would be indexed.
- Drug or Disease with score would be fetched using a Data API spec created . and This query to DB would be on demand when a user on UI would select either a drug or disease. In turn , request and response would be over Data API to get data from DB storing flattened records.
- The unique list of drugs or diseases would be rendered on UI at the time of page load for users to select the primary field and trigger data fetch

 ## 1. Database Choices and Decisions

### Context

The project requires a database that supports efficient querying of drug or disease data with minimal latency and cost-efficiency at scale. During the Minimum Viable Product (MVP) phase, we anticipate a moderate load of around 10,000 requests per day. However, as the application scales, we expect this to increase to millions of queries per day. Our architecture must therefore accommodate both immediate requirements and anticipated future growth.


### Alternatives Considered

- **Google Cloud SQL (PostgreSQL)** offers a reliable, managed database service suitable for our initial workloads. It is cost-effective, balances scalability, and supports complex queries needed for drug and disease data.
- **Google Cloud Memory Store (Redis)** is designed for high-performance data access, making it an ideal choice as our workload increases. Implementing a caching layer will minimize direct database hits and further reduce latency, ensuring scalability without substantial operational overhead.

### Decision

We will use **Google Cloud SQL with PostgreSQL** as our primary database for the MVP phase. This relational database solution provides adequate performance and scalability for moderate workloads while also allowing for straightforward data modeling, indexing, and querying.

For future scaling, we will introduce **Google Cloud Memory Store (Redis)** as a caching layer. This will provide fast in-memory caching capabilities, reducing the load on PostgreSQL and improving query response times.


### References

- [Google Cloud SQL Documentation](https://cloud.google.com/sql)
- [Google Cloud Memory Store Documentation](https://cloud.google.com/memorystore/docs)



## 2. Data API Choices and Decisions


### Context

Our application requires a reliable, maintainable API framework for performing CRUD operations on drug and disease data. Key considerations include ease of integration, extensibility, and support for relational database interactions. To maintain a consistent technology stack, we prefer a solution compatible with our team's expertise and existing ecosystem.


### Alternatives Considered

1. **FastAPI** is a Python-based framework, making it compatible with our tech stack and Python's extensive libraries for data science, which may be beneficial in future developments.
    - It offers asynchronous request handling, which improves performance, and has built-in support for RESTful API development, making it suitable for CRUD operations.
    - FastAPI’s SQLModel allows seamless interaction with relational databases, specifically designed for use with SQL databases like PostgreSQL.


2. **Node.js with Express**
   - **Pros**: High performance and robust ecosystem.
   - **Cons**: Adds additional language and framework complexity to our current Python stack.

3. **Spring Boot (Java)**
   - **Pros**: Extensive support for enterprise-grade applications.
   - **Cons**: Higher complexity and not aligned with our team's Python-based workflow.

### Decision

We will use **FastAPI** as the API framework for implementing CRUD operations. FastAPI is a modern, Python-based web framework that offers high performance, asynchronous request handling, and straightforward integration with SQLModel for database interaction.


### References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SQLModel Documentation](https://sqlmodel.tiangolo.com/)


   
