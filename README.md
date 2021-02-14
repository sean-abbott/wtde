# Installation and running
(Note it doesn't work right now)
You can use pip to install it (or pipx if you wanted it on your path), i.e., `pip install .`
You can then run `wtde` and what works, work.

## One-off runs
To run against a single directory (one of the examples, perhaps), you can run 
```
wtde extract examples/air
```

If you've downloaded the images you'll get a short readout with the fact it was an air game, what the map was, and whether I won or lost.

## Watch a directory
To watch a directory, you can use
```
wtde watch /tmp/directory
```

This will watch the directory. Currently watching the directory does nothing other than printing the files when there's at least 3 of them.

## Developer Setup
The examples screenshots are large, so I threw out on backblaze. You can either manually download them and unzip them from [here](https://f002.backblazeb2.com/file/SeansPublicFileShares/wtde_examples.tar.gz) or just run `scripts/get_examples.sh`, which I haven't tested

Then, to get the packages into an easily importable form for ipython or jupyter, you can `pip install -e .` and and you can re-import whenever you make changes.
