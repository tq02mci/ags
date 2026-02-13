# 阿里云 A股量化交易系统 - Terraform 配置

terraform {
  required_providers {
    alicloud = {
      source  = "aliyun/alicloud"
      version = "~> 1.209"
    }
  }
  required_version = ">= 1.0"

  # 远程状态存储（可选）
  # backend "oss" {
  #   bucket = "your-terraform-state-bucket"
  #   prefix = "a-stock-quant"
  #   region = "cn-shanghai"
  # }
}

provider "alicloud" {
  region = var.region
}

# 变量定义
variable "region" {
  description = "阿里云区域"
  type        = string
  default     = "cn-shanghai"
}

variable "project_name" {
  description = "项目名称"
  type        = string
  default     = "a-stock-quant"
}

variable "db_password" {
  description = "数据库密码"
  type        = string
  sensitive   = true
}

# 获取可用区信息
data "alicloud_zones" "default" {
  available_disk_category     = "cloud_efficiency"
  available_resource_creation = "VSwitch"
}

# 获取最新ECS镜像
data "alicloud_images" "ubuntu" {
  most_recent = true
  name_regex  = "^ubuntu_22"
  owners      = "system"
}

# 获取实例规格
data "alicloud_instance_types" "default" {
  availability_zone = data.alicloud_zones.default.zones[0].id
  cpu_core_count    = 2
  memory_size       = 4
}

# 输出信息
output "vpc_id" {
  value = alicloud_vpc.main.id
}

output "rds_endpoint" {
  value = alicloud_db_instance.main.connection_string
}

output "ecs_public_ip" {
  value = alicloud_instance.main.public_ip
}

output "oss_bucket" {
  value = alicloud_oss_bucket.data.bucket
}
