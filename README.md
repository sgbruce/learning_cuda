## Setup Instructions to run AVX512
To run avx512 instructions, you need to setup an EC2 instance.
Go to the EC2 page on the AWS console and select a stable LTS Ubuntu version, 
and for the server type choose c6i.xlarge. Choose a small (default) amount of storage 
and generate a key pair to use to connect if necessary, then initialize. Clone this repo into the 
launched container (once SSHed), and run the ec2setup script. You should then be able to compile and run 
avx512 instructions.