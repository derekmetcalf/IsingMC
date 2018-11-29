import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.io import imread
from skimage.io import imsave
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage import img_as_uint

def initialize_lattice(N_1,N_2,mode):
    """
    initializes an NxN np array lattice with spins +-1
    """
    lattice = np.zeros((N_1,N_2))
    if mode == "random":
        for i in range(N_1):
    	    for j in range(N_2):
    	        rand = np.random.randint(2)
    	        if rand == 1:
    	            lattice[i,j] = 1
    	        else:
     	            lattice[i,j] = -1
    
    elif mode == "up":
        for i in range(N_1):
    	    for j in range(N_2):
                lattice[i,j] = 1 

    elif mode == "down":
        for i in range(N_1):
    	    for j in range(N_2):
                lattice[i,j] = -1
    
    elif mode == "JGM":
        #for this mode, N1 must be 253 and N2 199
        img = imread("./jgm.jpeg")
        img_gray = rgb2gray(img)
        thresh = threshold_otsu(img_gray)
        lattice = np.array(lattice,dtype = np.int16)
        for i in range(253):
            for j in range(199):
                if img_gray[i,j]>thresh:
                    lattice[i,j] = 1
                else:
                    lattice[i,j] = -1
    else:
        print("Invalid initialization mode declared")
        quit()
    return lattice
    
def add_movie_frame(lattice, frame):
    plt.cla()
    plt.figure(figsize=(30, 15))
    plt.imshow(lattice, cmap='hot')
    fname = '_tmp%03d.png' %frame
    plt.savefig(fname)
    plt.close()
    return fname

def get_energy_diff(J, site_init_spin, neighbor_spin):
    """
    computes difference in energy when changing one spin in a 2D lattice
    """
    return

def get_neighbor_spins(row, column, lattice, N_1, N_2):
    """
    gathers neighbor spins, indexed from 12 o clock clockwise
    """
    if row!=0:
        s_0 = lattice[row-1][column]
    elif row==0:
        s_0 = lattice[N_1-1][column]
    if column!=N_2-1:
        s_1 = lattice[row][column+1]
    elif column==N_2-1:
        s_1 = lattice[row][0]
    if row!=N_1-1:
        s_2 = lattice[row+1][column]
    elif row==N_1-1:
        s_2 = lattice[0][column]
    if column!=0:
        s_3 = lattice[row][column-1]
    elif column==0:
        s_3 = lattice[row][N_2-1]
    return (s_0 + s_1 + s_2 + s_3)

def MC_move(lattice,beta_J,H):
    """
    """
    i_x = np.random.randint(N_1)
    i_y = np.random.randint(N_2)
    attempt_spin = lattice[i_x,i_y]*-1
    spin_sum = get_neighbor_spins(i_x, i_y, lattice, N_1, N_2)
    ext_field_en_diff = -2*lattice[i_x,i_y]*H
    energy_diff = 2*beta_J*lattice[i_x,i_y]*spin_sum + ext_field_en_diff
    if energy_diff <= 0 or np.random.rand(1) <= np.exp(-energy_diff):
        lattice[i_x,i_y] = attempt_spin
    return lattice

def MC_traj(N_1, N_2, N_steps, beta_J, H, frames, init_mode, gif_name="no_gif"):
    images = []
    lattice = initialize_lattice(N_1, N_2, init_mode)
    MC_mag = []
    MC_mag_sq = []
    MC_energy = []
    MC_energy_sq = []
    with imageio.get_writer('./%s.gif'%(gif_name), mode='I') as writer: 
        count = 0
        cycle = 0
        frame = 0
        for i in range(N_steps):
            count += 1
            MC_move(lattice, beta_J, H)
            if gif_name != "no_gif":
                if i % np.ceil(N_steps/frames) == 0:
                    frame += 1
                    fname = add_movie_frame(lattice,frame)
                    image = imageio.imread(fname)
                    writer.append_data(image)
            if count % N_1*N_2 == 0:
                cycle +=1

                #equilibration condition to start computing properties
                if cycle > 0.5*N_steps/N_1/N_2:
                    mag, mag_sq, energy, energy_sq = MC_props(lattice, N_1, N_2, beta_J, H)
                    MC_mag.append(mag)
                    MC_mag_sq.append(mag_sq)
                    MC_energy.append(energy)
                    MC_energy_sq.append(energy_sq)
    mean_mag = np.average(MC_mag)
    mean_mag_sq = np.average(MC_mag_sq)
    mean_energy = np.average(MC_energy)
    mean_energy_sq = np.average(MC_energy_sq)
    return mean_mag, mean_mag_sq, mean_energy, mean_energy_sq

def analytic_mag(beta_J):
    onsager_mag = (1-(np.sinh(2*beta_J)**-4))**(1/8)
    return onsager_mag

def MC_props(lattice, N_1, N_2, beta_J, H):
    mag = np.sum(lattice)/N_1/N_2
    mag_sq = (np.sum(lattice)/N_1/N_2)**2
    spin_sums = 0
    lattice_spins = 0
    for i in range(N_1):
        for j in range(N_2):
            spin_sums += get_neighbor_spins(i, j, lattice, N_1, N_2)
            lattice_spins += lattice[i,j]
    energy = (-beta_J * spin_sums - H*lattice_spins)/N_1/N_2 
    energy_sq = energy ** 2
    return mag, mag_sq, energy, energy_sq

def prop_v_temperature(N_1, N_2, N_cycles, N_steps, H, points, init_mode):
    beta_vec = np.zeros(points)
    mag_vec = np.zeros(points)
    mag_sq_vec =np.zeros(points) 
    energy_vec = np.zeros(points)
    energy_sq_vec = np.zeros(points)
    onsager_mag_vec = np.zeros(points)
    for i in range(points):
        beta_vec[i] = 0.49 + (0.51-0.49)*i/points
        mag_vec[i], mag_sq_vec[i], energy_vec[i], energy_sq_vec[i] = MC_traj(
                            N_1, N_2, N_steps, beta_vec[i], H, 100, init_mode)
        print(mag_vec[i]) 
        print(mag_sq_vec[i]) 
        print(energy_vec[i]) 
        print(energy_sq_vec[i])
    return beta_vec, mag_vec, mag_sq_vec, energy_vec, energy_sq_vec

def get_fluctuations(mean_mag, mean_mag_sq, mean_energy, mean_energy_sq):
    sigma_M = np.sqrt(mean_mag_sq - mean_mag**2)
    sigma_E = np.sqrt(mean_energy_sq - mean_mag**2)
    return sigma_M, sigma_E

def get_heat_capacities(sigma_M, sigma_E, beta_J):
   C_V_class = sigma_E*beta_J**2
   mag1, magsq1, en1, ensq1 = MC_traj(20, 20, 1000000, (beta_J-0.02), 0, 100, "up")
   mag2, magsq2, en2, ensq2 = MC_traj(20, 20, 1000000, (beta_J+0.02), 0, 100, "up")
   C_V_num = abs(en2-en1)/abs(1/(beta_J+0.02)-1/(beta_J-0.02))
   return C_V_class, C_V_num

N_1 = 20
N_2 = 20
N_cycles = 2500
N_steps = N_cycles * N_1 * N_2
beta_J = 0.5
H = 0.5
init_mode = "up"
frames = 100
attempt_prob = 1/(N_1*N_2)

print("Analytic avg mag: %s"%(analytic_mag(beta_J)))
mean_mag, mean_mag_sq, mean_energy, mean_energy_sq = MC_traj(
                            N_1, N_2, N_steps,beta_J, H, frames,init_mode)
print("Monte Carlo avg mag: %s"%mean_mag)
print("Monte Carlo avg mag^2: %s"%mean_mag_sq)
print("Monte Carlo avg energy: %s"%mean_energy)
print("Monte Carlo avg energy^2: %s"%mean_energy_sq)


beta_J, mag, mag_sq, energy, energy_sq = prop_v_temperature(
                            N_1, N_2, N_cycles, N_steps, H, 2, init_mode)

mag_fluc = np.zeros(len(mag))
energy_fluc = np.zeros(len(energy))
heat_cap = np.zeros(len(energy))

for i in range(len(mag)):
    mag_fluc[i], energy_fluc[i] = get_fluctuations(mag[i], mag_sq[i], energy[i], energy_sq[i])
    heat_cap[i] = get_heat_capacities(mag_fluc[i], energy_fluc[i], beta_J[i])

onsager_beta_J = np.linspace(0.4,0.55,500)
onsager_mag = np.zeros(len(onsager_beta_J))
for i in range(len(onsager_beta_J)):
    onsager_mag[i] = analytic_mag(onsager_beta_J[i])
plt.plot(beta_J, mag, 'bs', onsager_beta_J, onsager_mag, 'r--')
plt.xlabel("Beta*J")
plt.ylabel("<M>")
monte_carlo = mpatches.Patch(color='blue', label="Monte Carlo")
onsager = mpatches.Patch(color='red', label="Onsager")
plt.legend(handles=[monte_carlo, onsager])
plt.show()
plt.close()

plt.plot(beta_J, mag_fluc, 'bs')
plt.xlabel("Beta*J")
plt.ylabel("Magnetization Fluctuations")
plt.show()
plt.close()


plt.plot(beta_J, energy_fluc, 'bs')
plt.xlabel("Beta*J")
plt.ylabel("Energy Fluctuations")
plt.show()
plt.close()

plt.plot(beta_J, heat_cap, 'r--')
plt.xlabel("Beta*J")
plt.ylabel("Heat Capacity Cv/J")
plt.show()
plt.close()
