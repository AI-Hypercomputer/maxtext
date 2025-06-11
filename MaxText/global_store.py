# MaxText/global_store.py
"""Global state management for MaxText from_pretrained functionality.

This module provides thread-safe global storage for mesh and init_rng,
maintaining compatibility with JAX's functional programming principles.
"""

import threading
from typing import Optional, Dict, Any, Tuple
from contextlib import contextmanager
import jax
from jax import random
from jax.sharding import Mesh
import logging

logger = logging.getLogger(__name__)


class MaxTextGlobalStore:
    """Thread-safe global store for MaxText mesh and initialization state.
    
    This store enables the from_pretrained function to return only the model
    while preserving access to mesh and init_rng throughout the training pipeline.
    """
    
    def __init__(self):
        self._lock = threading.RLock()
        self._mesh: Optional[Mesh] = None
        self._init_rng: Optional[random.PRNGKey] = None
        self.checkpoint_manager = None
        self.learning_rate_schedule = None 
        self.tx = None
        self._is_initialized = False
    
    def store_training_supporters(
        self, 
        mesh: Mesh, 
        init_rng: random.PRNGKey,
        checkpoint_manager,
        learning_rate_schedule, 
        tx,
    ) -> None:
        """Store training supporters like mesh, init_rng, checkpoint manager, learning_rate_schedule and optimizer
          globally.
        
        Args:
            mesh: JAX Mesh object for distributed computation
            init_rng: JAX PRNGKey for initialization
            checkpoint_manager: Checkpoint manager for saving/loading model state
            learning_rate_schedule: Learning rate schedule for training
            tx: Optimizer state for training
        """
        with self._lock:
            self._mesh = mesh
            self._init_rng = init_rng
            self.checkpoint_manager = checkpoint_manager
            self.learning_rate_schedule = learning_rate_schedule
            self.tx = tx
            self._is_initialized = True
            logger.info(f"Stored mesh, init_rng, checkpoint_manager, learning_rate_schedule, tx in global store" )
    
    def get_mesh(self) -> Optional[Mesh]:
        """Retrieve the stored mesh."""
        with self._lock:
            return self._mesh
    
    def get_init_rng(self) -> Optional[random.PRNGKey]:
        """Retrieve the stored init_rng."""
        with self._lock:
            return self._init_rng
    def get_checkpoint_manager(self):
        """Retrieve the stored checkpoint manager."""
        with self._lock:
            return self.checkpoint_manager
    def get_learning_rate_schedule(self):
        """Retrieve the stored learning rate schedule."""
        with self._lock:
            return self.learning_rate_schedule

    def get_tx(self):
        """Retrieve the stored optimizer state."""
        with self._lock:
            return self.tx

    def get_mesh_and_init_rng(self) -> Tuple[Optional[Mesh], Optional[random.PRNGKey]]:
        """Retrieve both mesh and init_rng atomically."""
        with self._lock:
            return self._mesh, self._init_rng

    def get_training_supporters(self):
        """Retrieve both mesh and init_rng atomically."""
        with self._lock:
            return self._mesh, self._init_rng, self.checkpoint_manager, self.learning_rate_schedule, self.tx
   
    def is_initialized(self) -> bool:
        """Check if the store has been initialized with values."""
        with self._lock:
            return self._is_initialized
    
    def clear(self) -> None:
        """Clear all stored state."""
        with self._lock:
            self._mesh = None
            self._init_rng = None
            self.checkpoint_manager = None
            self.learning_rate_schedule = None
            self.tx = None
            self._is_initialized = False
            logger.info("Cleared global state")

    @contextmanager
    def temporary_state(
        self, 
        mesh: Mesh, 
        init_rng: random.PRNGKey,
        checkpoint_manager,
        learning_rate_schedule,
        tx
    ):
        """Context manager for temporary state changes.
        
        Useful for testing or isolated operations that need different
        mesh/rng configurations without affecting global state.
        
        Example:
            with global_store.temporary_state(test_mesh, test_rng):
                # Operations here use test_mesh and test_rng
                model = from_pretrained(...)
            # Original state is restored here
        """
        with self._lock:
            # Save current state
            old_mesh = self._mesh
            old_init_rng = self._init_rng
            old_is_initialized = self._is_initialized
            old_checkpoint_manager = self.checkpoint_manager
            old_learning_rate_schedule = self.learning_rate_schedule
            old_tx = self.tx
            
            # Set temporary state
            self._mesh = mesh
            self._init_rng = init_rng
            self._is_initialized = True
            self.checkpoint_manager = checkpoint_manager
            self.learning_rate_schedule = learning_rate_schedule
            self.tx = tx
            logger.info("Temporarily set global state for context manager")
        try:
            yield self
        finally:
            with self._lock:
                # Restore original state
                self._mesh = old_mesh
                self._init_rng = old_init_rng
                self._is_initialized = old_is_initialized
                self.checkpoint_manager = old_checkpoint_manager
                self.learning_rate_schedule = old_learning_rate_schedule
                self.tx = old_tx


# Global singleton instance
_global_store = MaxTextGlobalStore()


def get_global_store() -> MaxTextGlobalStore:
    """Get the global MaxText store singleton.
    
    This function provides access to the global store that maintains
    mesh and init_rng state across the MaxText training pipeline.
    
    Returns:
        MaxTextGlobalStore: The global store instance
    """
    return _global_store


def store_global_state(mesh: Mesh, init_rng: random.PRNGKey, checkpoint_manager, learning_rate_schedule, tx) -> None:
    """Convenience function to store state in the global store.
    
    Args:
        mesh: JAX Mesh object for distributed computation
        init_rng: JAX PRNGKey for initialization
    """
    get_global_store().store_training_supporters(mesh, init_rng, 
                                                 checkpoint_manager,learning_rate_schedule, tx)


def get_global_mesh() -> Optional[Mesh]:
    """Convenience function to get the global mesh."""
    return get_global_store().get_mesh()


def get_global_init_rng() -> Optional[random.PRNGKey]:
    """Convenience function to get the global init_rng."""
    return get_global_store().get_init_rng()


def get_global_checkpoint_manager() -> Any:
    """Convenience function to get the global checkpoint manager."""
    return get_global_store().checkpoint_manager

def get_global_learning_rate_schedule() -> Any:
    """Convenience function to get the global learning rate schedule."""
    return get_global_store().learning_rate_schedule

def get_global_tx() -> Any:
    """Convenience function to get the global optimizer state."""
    return get_global_store().tx

def get_global_mesh_and_init_rng() -> Tuple[Optional[Mesh], Optional[random.PRNGKey]]:
    """Convenience function to get both global mesh and init_rng."""
    return get_global_store().get_mesh_and_init_rng()

def clear_global_state() -> None:
    """Convenience function to clear global state."""
    get_global_store().clear()


# For backward compatibility or alternative naming preferences
global_store = get_global_store