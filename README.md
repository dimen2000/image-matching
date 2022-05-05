### Demo for image-matching network from [SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork)

#### Option 1 run with docker

##### Requirements
* Docker

##### Steps

Build image with docker

```bash
$ docker build --tag image-matching .
```
Run image

```bash
$ docker run --name img-match-doc image-matching
```

Copy result image to local directory

```bash
$ docker cp img-match-doc:/result .
```

Remove docker image

```bash
$ docker rm img-match-doc
```

#### Option 2 run with python via conda

##### Requirements
* Python
* Conda

Create superglue enviropment for conda

```bash
$ conda env create -f environment.yml
```

Activate superglue enviropment

```bash
$ conda activate superglue
```

Run script

```bash
$ python test.py
```

#### Option 3 run with python via pipenv

##### Requirements
* Python
* pipenv

Create superglue enviropment

```bash
$ pipenv shell
```

Install requirements

```bash
$ pip install -r requirements.txt
```

Run script

```bash
$ python test.py
```
