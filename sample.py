#from numba.core.errors import LoweringError
# Gives random spin configuration

from numba import njit,prange
def randomstate(N):
    return 2 * np.random.randint(2, size=(N, N)) - 1

# Performs one Metropolis step
@njit(parallel=True) # simple parallel for loop
def metropolis_step(lattice, T,H_field):
    for _ in prange(N * N):
        x, y = np.random.randint(0, N), np.random.randint(0, N)
        deltaE = 2 * lattice[x, y] * (lattice[(x + 1) % N, y] + lattice[x, (y + 1) % N] + # Periodic boundary conditions
                       lattice[(x - 1) % N, y] + lattice[x, (y - 1) % N])+2*lattice[x, y]*H_field
        if deltaE < 0:
            lattice[x, y] *= -1
        elif np.random.rand() < np.exp(-deltaE / T):
            lattice[x, y] *= -1

# Saves data
def save(object, filename):
    with open(filename + '.pickle', 'wb') as f:
        pickle.dump(object, f)


# Setting up simulation parameters
Hc=2.0
N = 20
Tc = 2.269
MCsteps = 5000
EQsteps = 2000
ntemp = 3000

T=1
H_mosbat = np.linspace(4, -4, ntemp)
magnetization = []

# Initializing spins and labels
spins, labels = np.zeros((0, N * N)), np.zeros((0, 4))
low,c_between1,c_between2,high = np.array([1,0,0,0]), np.array([0,1,0,0]), np.array([0, 0,1,0]), np.array([0, 0,0,1])

# Perform simulation
attice = randomstate(N)
for index,Hfield in enumerate(H_mosbat):
    tmp = []
    lattice = lattice.copy()
    # Equilibrate spin lattice
    for _ in range(EQsteps):
        metropolis_step(lattice,T, Hfield)
    # Loop over spin configurations to collect data
    for mc in range(MCsteps):
        if mc % 200 == 0:
            tmp.append(np.sum(lattice))
        metropolis_step(lattice, T,Hfield)
#    spins = np.vstack((spins, lattice.ravel()))
    magnetization.append(np.mean(tmp) / (N * N))
    # Append correct label corresponding to current spin configuration
    if (Hfield <= -0.5) & ((np.sum(lattice) / (N * N)) < -0.98)  :
        labels = np.vstack((labels,low))
        spins = np.vstack((spins, lattice.ravel()))
    elif (-0.5< Hfield < 0.5) &  (np.sum(lattice) / (N * N)<-0.98):
        labels = np.vstack((labels, c_between1))
        spins = np.vstack((spins, lattice.ravel()))
    elif (-0.5< Hfield < 0.5) & (np.sum(lattice) / (N * N)>0.98):
        labels = np.vstack((labels, c_between2))
        spins = np.vstack((spins, lattice.ravel()))
    elif (Hfield >= 0.5) & ((np.sum(lattice) / (N * N)) > 0.98):      
        labels = np.vstack((labels, high))
        spins = np.vstack((spins, lattice.ravel()))

    print('{} out of {} temperature steps'.format(index, len(H_mosbat)))

# Save data
save(0.5 * (spins + 1), 'traindata_H_mosbat_rev'), save(labels, 'trainlabel_H_mosbat_rev'), save(Hfield ,'maydan_H_mosbat_rev')                                                                         
print("saved data!")
#####################################manfi#################################
H_manfi = np.linspace(-4, 4, ntemp)
magnetization2 = []

# Initializing spins and labels
spins, labels = np.zeros((0, N * N)), np.zeros((0, 4))
low,c_between1,c_between2,high = np.array([1,0,0,0]), np.array([0,1,0,0]), np.array([0, 0,1,0]), np.array([0, 0,0,1])

# Perform simulation
lattice = randomstate(N)
for index,Hfield in enumerate(H_manfi):
    tmp = []
    lattice = lattice.copy()
    # Equilibrate spin lattice
    for _ in range(EQsteps):
        metropolis_step(lattice,T, Hfield)
    # Loop over spin configurations to collect data
    for mc in range(MCsteps):
        if mc % 200 == 0:
            tmp.append(np.sum(lattice))
        metropolis_step(lattice, T,Hfield)
#    spins = np.vstack((spins, lattice.ravel()))
    magnetization2.append(np.mean(tmp) / (N * N))
    # Append correct label corresponding to current spin configuration
    if (Hfield <= -0.5) & ((np.sum(lattice) / (N * N)) < -0.98)  :
        labels = np.vstack((labels,low))
        spins = np.vstack((spins, lattice.ravel()))
    elif (-0.5< Hfield < 0.5) &  (np.sum(lattice) / (N * N)<-0.98):
        labels = np.vstack((labels, c_between1))
        spins = np.vstack((spins, lattice.ravel()))
    elif (-0.5< Hfield < 0.5) & (np.sum(lattice) / (N * N)>0.98):
        labels = np.vstack((labels, c_between2))
        spins = np.vstack((spins, lattice.ravel()))
    elif (Hfield >= 0.5) & ((np.sum(lattice) / (N * N)) > 0.98):      
        labels = np.vstack((labels, high))
        spins = np.vstack((spins, lattice.ravel()))

    print('{} out of {} temperature steps'.format(index, len(H_mosbat)))

# Save data
save(0.5 * (spins + 1), 'traindata_H_manfi'), save(labels, 'trainlabel_H_manfi'), save(Hfield ,'H_manfi')                                                                         
print("saved data!")


# Plot Monte Carlo simulation of magnetization
plt.scatter(np.array(H_manfi), np.array(magnetization2), c ="blue",
            linewidths = 3,
            #marker ="o",
            edgecolor ="blue",
            s = 80)
plt.scatter(np.array(H_mosbat), np.array(magnetization), c ="green",
            linewidths = 1,
          #  marker ="^",
            edgecolor ="green",
            s = 40)
plt.xlabel("External field", fontsize=15)
plt.ylabel("Magnetization ", fontsize=15)
plt.grid()
plt.savefig('magnetization_MC.png')
plt.show()


