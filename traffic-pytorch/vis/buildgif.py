import imageio 

with imageio.get_writer('bj.gif', mode='I') as writer:
    for filename in ['imgs/{}.png'.format(i) for i in range(50)]:
        image = imageio.imread(filename)
        writer.append_data(image)