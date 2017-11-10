def generate_arrow(i, j, v0, v1, p0, p1):
	"""
		i := limb[0]
		j := limb[1]
		v0 := (x0, y0)
		v1 := (x1, y1)
		p0 := rankmat[i][j]
		p1 := rankmat[j][i]
		# arrow from z_i >= z_j [higher -> lower] by default
	"""
	pStart, pEnd = (v0, v1) if p0 >= p1 else (v1, v0)
	return pStart, pEnd, confidence_color(p0, p1)


def confidence_color(p0, p1):
	doll = min(255, max(0, int(abs(p0-p1)*255)))
	return (doll, 24, 255-doll)
