"""FastAPI dependencies."""

from typing import Annotated
from fastapi import Depends
from app.config import get_settings, Settings

SettingsDep = Annotated[Settings, Depends(get_settings)]
