#----- Provider configuration (OpenStack) ------------------------------------------
terraform {
  required_providers {openstack = {source = "terraform-providers/openstack"} }
  required_version = ">= 0.13"
}
provider "openstack" {}

#----- Instance --------------------------------------------------------------------
resource "openstack_compute_instance_v2" "instance" {
  name        = "my_first_instance"
  image_name  = "NeCTAR CentOS 7 x86_64"
  flavor_name = "m3.small"
  key_pair    = "sam-master"
  security_groups = ["default","SSH"]
}
