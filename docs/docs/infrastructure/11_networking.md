---
title: Networking
---
TODO add network diagram showing hub/spoke setup of the various projects

The networking layer of our software engineering project is designed to strike a balance between minimal maintenance requirements and optimal control and security. To achieve this, we employ a hub-and-spoke architecture, where the central hub network provides a Virtual Private Cloud (VPC) to our spoke projects, which represent our working group projects.

To ensure segregation of development and production environments, we maintain two separate networks for each. This allows for effective isolation and management of our infrastructure.

The below graphic visualizes this hub/spoke setup
![](../assets/img/mtrx_network.drawio)

## Firewall Configuration

To facilitate secure communication between our networks and the outside world, we have defined firewall routes to permit the following traffic:

1. HTTP and HTTPS traffic for accessing web interfaces
1. SSH traffic for remote access to our instances and tunneling, only permitted on `dev`

This configuration enables secure and controlled access to our infrastructure, while maintaining the integrity of our network architecture.

## Terraform

All networking is configured in terraform and changes can be proposed through Pull Requests. 


