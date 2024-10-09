PLATFORM=linux
include Makefile.in.$(PLATFORM)

.PHONY: all exe doc clean realclean

exe: sph.x 
doc: main.pdf derivation.pdf
all: exe doc

# =======

sph.x: sph.o params.o state.o binhash.o interact.o leapfrog.o io_txt.o 
	$(CPP) $(CPPFLAGS) $^ -o $@ $(LIBS)

sph.o: sph.cpp params.hpp state.hpp binhash.hpp interact.hpp leapfrog.hpp io.hpp 

params.o: params.cpp params.hpp
state.o: state.cpp state.hpp binhash.hpp
binhash.o: binhash.cpp binhash.hpp state.hpp zmorton.hpp
interact.o: interact.cpp interact.hpp state.hpp params.hpp binhash.hpp
leapfrog.o: leapfrog.cpp leapfrog.hpp state.hpp params.hpp
io_txt.o: io_txt.cpp io.hpp

%.o: %.c
	$(CPP) -c $(CPPFLAGS) $(OPTFLAGS) $<

# =======
main.pdf: main.tex codes.tex
derivation.pdf: derivation.tex check_derivation.tex

codes.tex: params.hpp state.hpp interact.cpp binhash.hpp zmorton.hpp binhash.cpp \
		leapfrog.cpp sph.cpp params.cpp io_txt.cpp 
	dsbweb -o $@ -c $^

check_derivation.tex: check_derivation.m
	dsbweb -o $@ -m $^

%.pdf: %.tex
	pdflatex $<
	pdflatex $<

# =======
stripped:
	rm -rf strip
	mkdir -p strip
	for f in *.cpp ; do awk -f strip.awk $$f > strip/$$f; done
	for f in *.hpp ; do awk -f strip.awk $$f > strip/$$f; done
	cp Makefile* README.md strip
	cp check_derivation.m derivation.tex main.tex strip
	
# =======

clean:
	rm -f *~ *.o
	rm -f codes.tex check_derivation.tex
	rm -f main.log main.aux main.out main.toc
	rm -f derivation.log derivation.aux derivation.out

realclean: clean
	rm -f main.pdf derivation.pdf *.x run.out *.pov
