FROM stepankuzmin/pytorch-notebook

COPY . /home/jovyan/work

ENTRYPOINT [ "start-notebook.sh" ]
