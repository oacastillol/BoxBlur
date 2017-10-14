all:
	g++ blur-effect.c -o blur-effect `pkg-config --cflags --libs opencv`
