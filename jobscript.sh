# ask for eight tasks
#SBATCH --ntasks=8
 
# Ask for one node, use several nodes in case you need additional resources
#SBATCH --nodes=6
 
# ask for less than 4 GB memory per task=MPI rank
#SBATCH --mem-per-cpu=3900M   #M is the default and can therefore be omitted, but could also be K(ilo)|G(iga)|T(era)
 
# name the job
#SBATCH --job-name=DEEPLENSE

#SBATCH --time=00:30:30
 
### beginning of executable commands
### Change to working directory
cd ${HPCDIR}/home/GSoC-2022

### Execute your application
python3 main.py --dataset_name Model_I --no-cuda