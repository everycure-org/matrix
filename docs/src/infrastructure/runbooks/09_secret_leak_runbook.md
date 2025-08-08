# Secret Leak on GitHub Runbook

When a secret is leaked on GitHub, we need to fill out the following post-mortem template. It is
pre-filled with an actual incident that happened on 2025-08-06.

## 1. **Incident Summary**

- **Incident Title:** Secret Leak on GitHub Repository
- **Date/Time Detected:** 2025-08-06
- **Date/Time Resolved:** 2025-08-14
- **Reported By:** GitHub Actions & OpenAI automated detection
- **Affected Systems/Services:** Google Cloud Platform, OpenAI, Neo4J Dev Instance
- **Severity Level:** [Medium]
- **Public or Private Repo:** Public

## 2. **Timeline of Events**

| Time             | Event/Action              | Description                                                                                                              |
| ---------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| 2024-07-01 18:27 | First Secret committed    | We committed our first secret to our then private repository                                                             |
| Early 2025       | Git History Rewrite       | We performed a git history rewrite to clean up any IP and did a rough search for secrets                                 |
| 2025-08-06 10:00 | Open Sourcing of Repo     | We published the matrix repository, turning it public                                                                    |
| 2025-08-06 12:00 | Automated Scanning alerts | Automatic secret scanning on GH caught secrets in git history                                                            |
| 2025-08-06 17:00 | 9/10 secrets mitigated    | All but one secrets were revoked, none were used in our production systems anymore                                       |
| 2025-08-06 17:00 | New License Key requested | Remaining secret requires a new license key for us to rotate the old one, we will continue to use the old one until then |

## 3. **Root Cause Analysis**

- **Cause of the Leak:**  
  Human Error, missing guardrails in our commit hooks, we had not enabled the automatic secret
  scanning on GitHub because it was not part of our license. It was only allowed for public
  repositories.
- **Detection:**  
  Automated systems notified us about the leaked secrets once the repository was published.
- **Scope:**  
  10 secrets were reported, about 5 were actual secrets and one was a license key.

## 4. **Immediate Actions Taken**

- Revoked and rotated compromised secrets.
- Published/internal notified stakeholders as needed.
- Activated push protection on the repository.
- Added a pre-commit hook to check for secrets before committing.

## 5. **Impact**

- **Services Affected:** None
- **Downtime:** None
- **Data Exposed:** None
- **External Communication Required:** No

## 6. **Lessons Learned**

- We should have scanned for `sk-proj` in the history and we would have caught the OAI secrets
- A 'high entropy' string detector would have caught the license key

## 7. **Preventative Actions**

- Adopt/enforce secret scanning in git workflows.
- Implement pre-commit/pre-push hooks.

## 8. **Follow-up Actions**

- Action 1: Rotate the license key in the dev environment
