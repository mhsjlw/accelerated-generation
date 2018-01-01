import numpy
import math

from noise.cpu import CombinedNoise, OctaveNoise
# from noise import JavaRandom
# from noise.gpu import combined_noise_compute, octave_noise_compute

class NormalGenerator:
    def __init__(self, world_x, world_y, world_z, random):
        self.random = random
        self.world_x = world_x
        self.world_y = world_y
        self.world_z = world_z
        self.total_size = world_x * world_y * world_z

        self.flood_data = numpy.zeros(1048576, dtype=numpy.int32)
        self.data = bytearray(self.total_size)

    def compute(self):
        raise_noise_1 = CombinedNoise(OctaveNoise(8, self.random), OctaveNoise(8, self.random))
        raise_noise_2 = CombinedNoise(OctaveNoise(8, self.random), OctaveNoise(8, self.random))
        raise_octaves = OctaveNoise(6, self.random)
        flat_noise = numpy.zeros(self.world_x * self.world_z, dtype=numpy.int32)

        print('Building heightmap')
        for x in range(0, self.world_x):
            for z in range(0, self.world_z):
                val1 = raise_noise_1.compute((x * 1.3), (z * 1.3)) / 6 + -4
                val2 = raise_noise_2.compute((x * 1.3), (z * 1.3)) / 5 + 6

                if raise_octaves.compute(x, z) / 8 > 0:
                    val2 = val1

                noise = max(val1, val2) / 2
                if noise < 0:
                    noise *= 0.8

                flat_noise[x + z * self.world_x] = math.floor(noise)

        erode_noise_1 = CombinedNoise(OctaveNoise(8, self.random), OctaveNoise(8, self.random))
        erode_noise_2 = CombinedNoise(OctaveNoise(8, self.random), OctaveNoise(8, self.random))

        print('Building strata')
        for x in range(0, self.world_x):
            for z in range(0, self.world_z):
                val1 = erode_noise_1.compute((x << 1), (z << 1)) / 8
                val2 = 1 if erode_noise_2.compute((x << 1), (z << 1)) > 0 else 0

                if val1 > 2:
                    val = ((flat_noise[x + z * self.world_x] - val2) // 2 << 1) + val2
                    flat_noise[x + z * self.world_x] = val

        print('Planting grass')
        soil_noise = OctaveNoise(8, self.random)
        for x in range(0, self.world_x):
            for z in range(0, self.world_z):
                val = math.floor(soil_noise.compute(x, z) // 24) - 4
                base = flat_noise[x + z * self.world_x] + ((self.world_y // 2) - 1)
                height = base + val
                flat_noise[x + z * self.world_x] = max(base, height)

                if flat_noise[x + z * self.world_x] > self.world_y - 2:
                    flat_noise[x + z * self.world_x] = self.world_y - 2

                if flat_noise[x + z * self.world_x] < 1:
                    flat_noise[x + z * self.world_x] = 1

                for y in range(0, self.world_y):
                    block = 0x00
                    if y <= base:
                        block = 0x03

                    if y <= height:
                        block = 0x01

                    if y == 0:
                        block = 0x10

                    self.data[(y * self.world_z + z) * self.world_x + x] = block

        print('Carving caves')
        caves = self.world_x * self.world_y * self.world_z // 256 // 64 << 1

        for cave in range(0, caves):
            baseX = self.random.next_float() * (self.world_x + 1)
            baseY = self.random.next_float() * (self.world_y + 1)
            baseZ = self.random.next_float() * (self.world_z + 1)

            total = math.floor((self.random.next_float() + self.random.next_float()) * 200.0)

            theta1 = self.random.next_float() * math.pi * 2
            theta1Mod = 0
            theta2 = self.random.next_float() * math.pi * 2
            theta2Mod = 0

            rad = self.random.next_float() * self.random.next_float()

            for count in range(0, total):
                baseX += math.sin(theta1) * math.cos(theta2)
                baseZ += math.cos(theta1) * math.cos(theta2)
                baseY += math.sin(theta2)

                theta1 += theta1Mod * 0.18
                theta1Mod = theta1Mod + (self.random.next_float() - self.random.next_float())
                theta2 = (theta2 + theta2Mod * 0.5) * 0.375
                theta2Mod = theta2Mod + (self.random.next_float() - self.random.next_float())

                if self.random.next_float() >= 0.25:
                    cx = baseX + (self.random.next_float() * 5 - 2) * 0.2
                    cy = baseY + (self.random.next_float() * 5 - 2) * 0.2
                    cz = baseZ + (self.random.next_float() * 5 - 2) * 0.2

                    radius = (self.world_y - cy) / self.world_y
                    radius = 1.2 + (radius * 3.5 + 1) * rad
                    radius = math.sin(count * math.pi / total) * radius

                    bx = math.floor(cx - radius)
                    while bx <= math.floor(cx + radius):
                        by = math.floor(cy - radius)
                        while by <= math.floor(cy + radius):
                            bz = math.floor(cz - radius)
                            while bz <= math.floor(cz + radius):
                                dx = bx - cx
                                dy = by - cy
                                dz = bz - cz

                                if dx * dx + dy * dy * 2 + dz * dz < radius * radius and bx >= 1 and by >= 1 and bz >= 1 and bx < self.world_x - 1 and by < self.world_y - 1 and bz < self.world_z - 1:
                                    key = (by * self.world_z + bz) * self.world_x + bx

                                    if self.data[key] == 0x01:
                                        self.data[key] = 0
                                bz += 1
                            by += 1
                        bx += 1

        print('Populating ores')
        self.populate_ore(0x10, 90, 1, self.data)
        self.populate_ore(0x0f, 70, 2, self.data)
        self.populate_ore(0x0e, 50, 3, self.data)
        print('Done populating ores')

        for x in range(0, self.world_x):
            index = self.world_y // 2 - 1
            depth_index = self.world_z - 1
            print('First!')
            self.flood(x, index, 0, 0x08, self.data)
            print('Second!')
            self.flood(x, index, depth_index, 0x08, self.data)

        for z in range(0, self.world_z):
            index = self.world_y // 2 - 1
            index2 = self.world_x - 1
            index3 = self.world_y // 2 - 1
            print('Third!')
            self.flood(0, index, z, 0x08, self.data)
            print('Fourth!')
            self.flood(index2, index3, z, 0x08, self.data)

        print('Starting flooding')
        waterFloods = self.world_x * self.world_z // 8000
        for flood in range(0, waterFloods):
            print('Calculating flooding')
            x = (math.floor(self.random.next_float() * (self.world_x + 1)))
            y = math.floor((self.world_y - 1) - 1 - (math.floor(self.random.next_float() * 3)))
            z = (math.floor(self.random.next_float() * (self.world_z + 1)))
            if self.data[(y * self.world_z + z) * self.world_x + x] == 0:
                print('Started a flood from up here!')
                self.flood(x, y, z, 0x08, self.data)

        print('Lava flooding')

        lavaFloods = self.world_x * self.world_z * self.world_y // 20000
        for flood in range(0, lavaFloods):
            x = (math.floor(self.random.next_float() * (self.world_x + 1)))
            y = math.floor(self.random.next_float() * self.random.next_float() * (((self.world_y - 1) - 3)) + 1)
            z = (math.floor(self.random.next_float() * (self.world_z + 1)))
            if self.data[(y * self.world_z + z) * self.world_x + x] == 0:
                self.flood(x, y, z, 0x0a, self.data)
        print('Done flooding')

        print('Growing')
        growNoise1 = OctaveNoise(8, self.random)
        growNoise2 = OctaveNoise(8, self.random)

        for x in range(0, self.world_x):
            for z in range(0, self.world_z):
                sandy = growNoise1.compute(x, z) > 8
                gravelWater = growNoise2.compute(x, z) > 12
                y = flat_noise[x + z * self.world_x]
                key = (y * self.world_z + z) * self.world_x + x
                block = self.data[((y + 1) * self.world_z + z) * self.world_x + x] & 255
                if (block == 0x08 or block == 0x09) and y <= self.world_z / 2 - 1 and gravelWater:
                    self.data[key] = 0x0d

                if block == 0:
                    id = 0x02
                    if y <= self.world_y / 2 - 1 and sandy:
                        id = 0x0c

                    self.data[key] = id

        flowers = self.world_x * self.world_z // 3000

        for flower in range(0, flowers):
            type = (math.floor(self.random.next_float() * 3))
            x = (math.floor(self.random.next_float() * (self.world_x + 1)))
            z = (math.floor(self.random.next_float() * (self.world_z + 1)))

            for xc in range(0, 10):
                fx = x
                fz = z

                for xc in range(0, 5):
                    fx += (math.floor(self.random.next_float() * 7)) - (math.floor(self.random.next_float() * 7))
                    fz += (math.floor(self.random.next_float() * 7)) - (math.floor(self.random.next_float() * 7))

                    if (type < 2 or (math.floor(self.random.next_float() * 5)) == 0) and fx >= 0 and fz >= 0 and fx < self.world_x and fz < self.world_z:
                        y = flat_noise[fx + fz * self.world_x] + 1
                        if (self.data[(y * self.world_z + fz) * self.world_x + fx] & 255) == 0:
                            key = (y * self.world_z + fz) * self.world_x + fx

                            if (self.data[((y - 1) * self.world_z + fz) * self.world_x + fx] & 255) == 0x02:
                                if type == 0:
                                    self.data[key] = 0x25
                                elif type == 1:
                                    self.data[key] = 0x26

        shrooms = self.world_x * self.world_y * self.world_z // 2000

        for shroom in range(0, shrooms):
            type = (math.floor(self.random.next_float() * 3))
            x = (math.floor(self.random.next_float() * (self.world_x + 1)))
            y = (math.floor(self.random.next_float() * (self.world_y + 1)))
            z = (math.floor(self.random.next_float() * (self.world_z + 1)))
            for xc in range(0, 20):
                mx = x
                my = y
                mz = z

                for zc in range(0, 5):
                    mx += (math.floor(self.random.next_float() * 7)) - (math.floor(self.random.next_float() * 7))
                    my += (math.floor(self.random.next_float() * 3)) - (math.floor(self.random.next_float() * 3))
                    mz += (math.floor(self.random.next_float() * 7)) - (math.floor(self.random.next_float() * 7))

                    if (type < 2 or (math.floor(self.random.next_float() * 5)) == 0) and mx >= 0 and mz >= 0 and my >= 1 and mx < self.world_x and mz < self.world_z and my < flat_noise[mx + mz * self.world_x] - 1 and (self.data[(my * self.world_z + mz) * self.world_x + mx] & 255) == 0:
                        key = (my * self.world_z + mz) * self.world_x + mx

                        if (self.data[((my - 1) * self.world_z + mz) * self.world_x + mx] & 255) == 0x01:
                            if type == 0:
                                self.data[key] = 0x27
                            elif type == 1:
                                self.data[key] = 0x28

        trees = self.world_x * self.world_z // 4000
        for tree in range(0, trees):
            x = (math.floor(self.random.next_float() * self.world_x))
            z = (math.floor(self.random.next_float() * self.world_z))
            for xc in range(0, 20):
                tx = x
                tz = z

                for zc in range(0, 20):
                    tx += (math.floor(self.random.next_float() * 7)) - (math.floor(self.random.next_float() * 7))
                    tz += (math.floor(self.random.next_float() * 7)) - (math.floor(self.random.next_float() * 7))

                    if (tx >= 0 and tz >= 0 and tx < self.world_x and tz < self.world_z):
                        y = flat_noise[tx + tz * self.world_x] + 1
                        if (math.floor(self.random.next_float() * 4)) == 0:
                            self.grow_tree(self.data, tx, y, tz)
        print('Done growing')

        return self.data

    def grow_tree(self, blocks, x, y, z):
        print('Growing a tree!')
        height = (math.floor(self.random.next_float() * 3)) + 4
        space_free = True

        by = y
        while by <= y + 1 + height:
            radius = 1
            if by == y:
                radius = 0

            if by >= y + 1 + height - 2:
                radius = 2

            bx = x - radius
            while bx <= x + radius and space_free:
                bz = z - radius
                while bz <= z + radius and space_free:
                    if bx >= 0 and by >= 0 and bz >= 0 and bx < self.world_x and by < self.world_y and bz < self.world_z:
                        if (blocks[(by * self.world_z + bz) * self.world_x + bx] & 255) != 0:
                            space_free = False
                        else:
                            space_free = False
                bx += 1
            by += 1

        if not space_free:
            return False
        elif (blocks[((y - 1) * self.world_z + z) * self.world_x + x] & 255) == 0x02 and y < self.world_y - height - 1:
            blocks[((y - 1) * self.world_z + z) * self.world_x + x] = 0x03

            ly = y - 3 + height
            while ly <= y:
                base_dist = ly - (y + height)
                radius = 1 - base_dist / 2

                lx = x - radius
                while lx <= x + radius:
                    xdist = lx - x

                    lz = z - radius
                    while lz <= z + radius:
                        zdist = lz - z

                        if math.abs(xdist) != radius or math.abs(zdist) != radius or (math.floor(self.random.next_float() * 2)) != 0 and base_dist != 0:
                            blocks[(ly * self.world_z + lz) * self.world_x + lx] = 0x12

                            if math.abs(xdist) != radius or math.abs(zdist) != radius or (math.floor(self.random.next_float() * 2)) != 0 and base_dist != 0:
                                blocks[((ly - 1) * self.world_z + lz) * self.world_x + lx] = 0x12

                        lz += 1
                    lx += 1
                ly += 1

            ly = 0
            while ly < height:
                blocks[((y + ly) * self.world_z + z) * self.world_x + x] = 0x11;
                ly += 1

            return True
        else:
            return False

    def populate_ore(self, id, chance, stage, data):
        print('Populating ores!')

        ores = self.world_x * self.world_y * self.world_z // 256 // 64 * chance // 100
        for ore in range(0, ores):
            bx = self.random.next_float() * self.world_x
            by = self.random.next_float() * self.world_y
            bz = self.random.next_float() * self.world_z

            total = math.floor((self.random.next_float() + self.random.next_float()) * 75 * chance // 100)
            theta_1 = self.random.next_float() * (math.pi * 2)
            theta_1_mod = 0
            theta_2 = self.random.next_float() * (math.pi * 2)
            theta_2_mod = 0

            for count in range(0, total):
                bx += math.sin(theta_1) * math.cos(theta_2)
                bz += math.cos(theta_1) * math.cos(theta_2)
                by += math.sin(theta_2)

                theta_1 += theta_1_mod * 0.2
                theta_1_mod = (theta_1_mod * 0.9) + (self.random.next_float() - self.random.next_float())
                theta_2 = (theta_2 + theta_2_mod * 0.5) * 0.5
                theta_2_mod = (theta_2_mod * 0.9) + (self.random.next_float() - self.random.next_float())
                radius = math.sin(count * math.pi / total) * chance / 100 + 1

                ox = math.floor(bx - radius)
                while ox <= math.floor(bx + radius):
                    oy = math.floor(by - radius)
                    while oy <= math.floor(by + radius):
                        oz = math.floor(bz - radius)
                        while oz <= math.floor(bz + radius):
                            dx = ox - bx
                            dy = oy - by
                            dz = oz - bz

                            if dx * dx + dy * dy * 2 + dz * dz < radius * radius and ox >= 1 and oy >= 1 and oz >= 1 and ox < self.world_x - 1 and oy < self.world_y - 1 and oz < self.world_z - 1:
                                key = (oy * self.world_z + oz) * self.world_x + ox

                                if self.data[key] == 0x01:
                                    self.data[key] = id
                            oz += 1
                        oy += 1
                    ox += 1

    def flood(self, x, y, z, id, blocks):
        print('Starting a flood from down here!')
        datas = [[]]
        xshift = 1
        zshift = 1

        while 1 << xshift < self.world_x:
            xshift += 1

        while 1 << zshift < self.world_z:
            zshift += 1

        maxX = self.world_x - 1
        maxZ = self.world_z - 1
        counter = 1

        self.flood_data[0] = ((y << zshift) + z << xshift) + x
        flatArea = self.world_x * self.world_z

        while counter > 0:
            counter -= 1
            tail = self.flood_data[counter]

            if counter == 0 and len(datas) > 0:
                index = len(datas) - 1
                del datas[index]
                counter = len(self.flood_data)

            zval = tail >> xshift & maxZ
            xval = tail >> xshift + zshift
            xmin = tail & maxX
            xmax = tail & maxX

            while xmin > 0 and blocks[tail - 1] == 0:
                xmin -= 1
                tail -= 1

            while xmax < self.world_x and blocks[tail + xmax - xmin] == 0:
                xmax += 1

            zMinComplete = False
            zMaxComplete = False
            xComplete = False

            while xmin < xmax:
                blocks[tail] = id
                air = False

                if zval > 0:
                    air = blocks[tail - self.world_x] == 0

                    if air and not zMinComplete:
                        if counter == len(self.flood_data):
                            datas.append(numpy.copy(self.flood_data))
                            self.flood_data = numpy.zeros(1048576, dtype=numpy.int32)
                            counter = 0

                        self.flood_data[counter] = tail - self.world_x
                        counter += 1

                    zMinComplete = air

                if zval < self.world_z - 1:
                    air = blocks[tail + self.world_x] == 0

                    if air and not zMaxComplete:
                        if counter == len(self.flood_data):
                            datas.append(numpy.copy(self.flood_data))
                            self.flood_data = numpy.zeros(1048576, dtype=numpy.int32)
                            counter = 0

                        self.flood_data[counter] = tail + self.world_x
                        counter += 1

                    zMaxComplete = air

                if xval > 0:
                    endId = blocks[tail - flatArea]

                    if (id == 0x0a or id == 0x0b) and (endId == 0x08 or endId == 0x09):
                        blocks[tail - flatArea] = 0x01

                    air = endId == 0

                    if air and not xComplete:
                        if counter == len(self.flood_data):
                            datas.append(numpy.copy(self.flood_data))
                            self.flood_data = numpy.zeros(1048576, dtype=numpy.int32)
                            counter = 0

                        self.flood_data[counter] = tail - flatArea
                        counter += 1

                    xComplete = air

                tail += 1
                xmin += 1
