# Neo4J certificates

Our cluster endpoint has SSL enabled, we've added a set of certificates to our local setup to mimick the behaviour and have this covered by an integration test. The relevant certifates have been pushed to the repository, though should someone wish to re-generate them, use the steps below.

```bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout certs/bolt/private.key -out certs/bolt/public.crt \
    -config certs/openssl.cnf
```

Neo4j expects the SSL files in a specific format, hence concat the private and public key into a single file.

```bash
cat certs/bolt/private.key certs/bolt/public.crt > certs/bolt/private_and_public.pem
```

Finally, remove the private key as it's no longer needed.

```bash
rm certs/bolt/private.key
```