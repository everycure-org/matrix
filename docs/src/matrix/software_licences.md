# Dealing with Licenses in our project

To make sure we do not use any software that gets us into trouble later on, we have
`trivy` scan for licenses of software we use. However, we don't yet have this implemented
in our CI. This will have to be added to our release pipeline when we build it. For now,
the Makefile in our `matrix` pipeline holds the command to scan for licenses.
