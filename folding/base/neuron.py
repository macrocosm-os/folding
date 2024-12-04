import openmm


class BaseFolding:
    def check_openmm_version(self):
        """
        A method that enforces that the OpenMM version that is running the version specified in the __OPENMM_VERSION_TAG__.
        """
        try:
            self.openmm_version = openmm.__version__

            if __OPENMM_VERSION_TAG__ != self.openmm_version:
                raise OpenMMException(
                    f"OpenMM version mismatch. Installed == {self.openmm_version}. Please install OpenMM {__OPENMM_VERSION_TAG__}.*"
                )

        except Exception as e:
            raise e

        logger.success(f"Running OpenMM version: {self.openmm_version}")

    def setup_wandb_logging(self):
        if os.path.isfile(f"{self.config.neuron.full_path}/wandb_ids.pkl"):
            self.wandb_ids = load_pkl(
                f"{self.config.neuron.full_path}/wandb_ids.pkl", "rb"
            )
        else:
            self.wandb_ids = {}

    def add_wandb_id(self, pdb_id: str, wandb_id: str):
        self.wandb_ids[pdb_id] = wandb_id
        write_pkl(self.wandb_ids, f"{self.config.neuron.full_path}/wandb_ids.pkl", "wb")

    def remove_wandb_id(self, pdb_id: str):
        self.wandb_ids.pop(pdb_id)
        write_pkl(self.wandb_ids, f"{self.config.neuron.full_path}/wandb_ids.pkl", "wb")
