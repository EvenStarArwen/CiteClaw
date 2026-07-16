"""Model pricing/catalog endpoint."""

from fastapi import APIRouter

from catalog import catalog_payload


router = APIRouter(prefix="/api/catalog", tags=["catalog"])


@router.get("")
async def get_catalog():
    return catalog_payload()
