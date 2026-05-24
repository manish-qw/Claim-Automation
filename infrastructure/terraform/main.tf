# infrastructure/terraform/main.tf
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE:
#     Terraform configuration for the AWS production infrastructure.
#
# WHAT GOES HERE (resources to define):
#
#     NETWORKING:
#         aws_vpc              — dedicated VPC for CLAIMOS (CIDR: 10.0.0.0/16)
#         aws_subnet           — private subnets for DB + Kafka, public for API
#         aws_security_group   — ingress/egress rules per service
#
#     COMPUTE:
#         aws_ecs_cluster      — ECS Fargate cluster for API + agent containers
#         aws_ecs_task_definition — task defs for api + agents services
#         aws_ecs_service      — running services with desired_count
#
#     DATABASE:
#         aws_db_instance      — RDS PostgreSQL 16 (Multi-AZ for production)
#                               Instance: db.t3.medium
#                               Encrypted at rest: true
#                               Backup retention: 7 days
#
#     MESSAGING:
#         aws_msk_cluster      — Amazon MSK (Managed Kafka)
#                               3 broker nodes (az spread)
#                               kafka_version = "3.5.1"
#
#     AUDIT:
#         aws_qldb_ledger      — QLDB ledger for immutable audit trail
#                               permissions_mode = "STANDARD"
#         aws_dynamodb_table   — DynamoDB table for audit DLQ fallback
#
#     STORAGE:
#         aws_s3_bucket        — Documents bucket (server-side encryption)
#         aws_s3_bucket        — ML models bucket (versioning enabled)
#
#     GRAPH DATABASE:
#         Note: Neo4j runs on a dedicated EC2 instance (not managed service).
#         aws_instance         — Neo4j EC2: t3.large, EBS 100GB
#
#     IAM:
#         aws_iam_role         — ECS task execution role
#         aws_iam_policy       — policy for QLDB, DynamoDB, S3, MSK access
#
#     VARIABLES (define in variables.tf):
#         aws_region, environment, db_password, vpc_cidr
#
#     OUTPUTS (define in outputs.tf):
#         api_endpoint, db_endpoint, kafka_brokers, qldb_ledger_name
# ─────────────────────────────────────────────────────────────────────────────
