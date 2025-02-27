from shiny import render, ui
from shiny.express import input

from shiny.express import ui
from shiny.ui import page_navbar
from shinyswatch import theme
from functools import partial

ui.page_opts(
    title="MedDataKit", theme=theme.flatly, 
    page_fn=partial(page_navbar, id="page", bg="primary", position="static"),
    fillable=True
)

with ui.nav_panel("Home"):
    
    with ui.layout_columns(col_widths=(2, 10)):
        
        # left panel
        with ui.card(style="border-width: 0px;"):
            with ui.accordion(id="acc", open="Section A"):  
                with ui.accordion_panel("Basic Usage"):  
                    "Section A content"

                with ui.accordion_panel("Dataset"):  
                    "Section B content"
        
        # right panel
        with ui.card(style="border-width: 0px;"):
            "Card 3"

with ui.nav_panel("Page 2"):
    "Page 2 content"