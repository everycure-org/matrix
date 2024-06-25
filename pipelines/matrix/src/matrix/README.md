# Local development and execution

Our codebase features code that allows for fully localized execution of the pipeline and its' auxiliary services using `docker-compose`. This guide assists you in setting it up.

Requirements

```
docker installation
docker-compose
java + don't forget to link
```

Run 

```
docker-compose up -d
wait for bootstrapping script to finsish
visit localhost to check if pipeline works
```

```
envs on fabricated data
run fabricator
run integration
```

NOTE: Everyone needs access to the Sa, dont forget to use the right gcp project