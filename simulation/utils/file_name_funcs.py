def extract_common(fnames,delimiter='_',ignore=['learnt','configs']):

	# remove extension
	fnames = [f.split('.')[0] for f in fnames]

	common = None
	for f in fnames:

		terms = f.split(delimiter)
		terms = [t for t in terms if not (t in ignore)]

		if common is None:
			common = terms
		else:
			common = [c for c in common if c in terms]

	if common is None:
		return ""
	return delimiter.join(common)

