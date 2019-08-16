# ModularityStudy
Code supporting the paper "Katz-Eigenvector Centrality Community Detection"

##### Dependencies
python2.7, jupyter, matplotlib, cython, numpy, pandas, scipy, networkx, zen

Dockerfile is included if you wish to use Docker.

To build the container image:

```
docker build -t <container-name> path/to/directory/holding/the/dockerfile
```

To run the notebooks:

```bash
docker run -it --rm -v path/to/files:/home -p 8888:8888 <containter-name> jupyter-notebook --ip 0.0.0.0 --no-browser
```

Then open `localhost:8888/tree` in your browser, navigate to `/home` and open the notebook.

To run the python scripts:

```
docker run -it --rm -v path/to/files:/home <container-name>
```

A python terminal will open. To run `amzn_product.py`:

```python
import os
os.chdir('/home')
import amzn_product
amzn_product.main()
```
