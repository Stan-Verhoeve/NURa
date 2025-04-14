from numpy import array


def readfile(filename):
    f = open(filename, "r")
    data = f.readlines()[3:]  # Skip first 3 lines
    nhalo = int(data[0])  # number of halos
    radius = []

    for line in data[1:]:
        if line[:-1] != "#":
            radius.append(float(line.split()[0]))

    radius = array(radius, dtype=float)
    f.close()
    return (
        radius,
        nhalo,
    )  # Return the virial radius for all the satellites in the file, and the number of halos
