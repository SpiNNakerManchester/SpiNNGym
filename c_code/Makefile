DIRS = breakout inverted_pendulum multi_arm_bandit

all: $(DIRS)
	@for d in $(DIRS); do $(MAKE) -C $$d || exit $$?; done

clean: $(DIRS)
	@for d in $(DIRS); do $(MAKE) -C $$d GNU=$(GNU) clean || exit $$?; done
    