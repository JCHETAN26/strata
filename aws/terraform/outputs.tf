output "region" {
  value = var.region
}

output "stage" {
  value = var.stage
}

output "ssh_user" {
  value = "ubuntu"
}

output "main_public_ip" {
  value = one(aws_instance.main[*].public_ip)
}

output "shard_public_ips" {
  value = aws_instance.shard[*].public_ip
}

output "shard_private_ips" {
  value = aws_instance.shard[*].private_ip
}

output "coordinator_public_ip" {
  value = one(aws_instance.coordinator[*].public_ip)
}

output "coordinator_private_ip" {
  value = one(aws_instance.coordinator[*].private_ip)
}

output "client_public_ip" {
  value = one(aws_instance.client[*].public_ip)
}

output "cluster_description" {
  description = "Recorded with the sharding results."
  value = var.stage == "cluster" ? format(
    "%d x %s shards, %s coordinator, %s client, %s, %s, cluster placement group, %s",
    var.shard_count, var.shard_instance_type, var.coordinator_instance_type,
    var.client_instance_type, var.availability_zone, var.market, data.aws_ami.ubuntu.name
  ) : null
}
