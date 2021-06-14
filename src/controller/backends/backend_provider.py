# -*-coding:utf-8-*-
import threading
from .keras.keras_backend import KerasBackend
from .pytorch.pytorch_backend import PytorchBackend


class BackendProvider:
    instance_lock = threading.Lock()

    def __init__(self):
        self.current_backend = None
        self.initialized_backends = dict()

    def __new__(cls):
        if not hasattr(BackendProvider, "instance"):
            with BackendProvider.instance_lock:
                if not hasattr(BackendProvider, "_instance"):
                    BackendProvider.instance = object.__new__(cls)
        return BackendProvider.instance

    def activate_backend(self, backend_name):
        if not self._is_initialized(backend_name):
            self.initialized_backends[backend_name] = self._initialize_backends(backend_name)
        self.current_backend = self.initialized_backends[backend_name]

    def get_current_backend(self):
        if self.current_backend is None:
            raise Exception("no backends installed")
        return self.current_backend

    def _is_initialized(self, backend_name):
        return self.initialized_backends.__contains__(backend_name)

    @staticmethod
    def _initialize_backends(backend_name):
        """If new backends are implemented, please add here."""
        if backend_name == "Keras":
            return KerasBackend()
        elif backend_name == "Pytorch":
            return PytorchBackend()
        else:
            raise Exception("illegal backend name")

    @staticmethod
    def name_of_available_backends():
        """If new backends are implemented, please add the
         name of this new backends here."""
        return list({"Keras", "Pytorch"})


BACKEND_PROVIDER = BackendProvider()
