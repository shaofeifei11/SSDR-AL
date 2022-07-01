

def get_sampler_args_str(sampler_args):
    if len(sampler_args) == 0:
        return ""

    sampler_name = ""
    for element in sampler_args:
        sampler_name = sampler_name + element + "-"
    return sampler_name[:-1]

def get_w(w):
    s = ""
    for key in w:
        s = s + ", " + key + "=" + str(w[key])
    return s
