use crate::{
    components::{footer::Footer, nav::Nav},
    pages::{
        architecture::ArchitecturePage, bench::BenchPage, home::HomePage, sponsor::SponsorPage,
    },
    theme,
};
use leptos::prelude::*;

#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub enum Page {
    #[default]
    Home,
    Architecture,
    Sponsor,
    Bench,
}

fn page_from_hash() -> Page {
    web_sys::window()
        .and_then(|w| w.location().hash().ok())
        .map(|h| {
            let s = h.trim_start_matches('#').trim_start_matches('/');
            match s {
                s if s.starts_with("architecture") => Page::Architecture,
                s if s.starts_with("sponsor") => Page::Sponsor,
                s if s.starts_with("bench") => Page::Bench,
                _ => Page::Home,
            }
        })
        .unwrap_or_default()
}

pub fn navigate(page: Page) {
    let hash = match page {
        Page::Home => "#/",
        Page::Architecture => "#/architecture",
        Page::Sponsor => "#/sponsor",
        Page::Bench => "#/bench",
    };
    if let Some(w) = web_sys::window() {
        let _ = w.location().set_hash(hash);
    }
}

#[component]
pub fn App() -> impl IntoView {
    let theme = RwSignal::new(theme::load_theme());
    let page = RwSignal::new(page_from_hash());

    provide_context(theme);
    provide_context(page);

    Effect::new(move |_| {
        theme::apply_theme(theme.get());
    });
    Effect::new(move |_| {
        navigate(page.get());
    });
    Effect::new(move |_| {
        let _ = page.get();
        if let Some(window) = web_sys::window() {
            if let Ok(func) =
                js_sys::Reflect::get(&window, &wasm_bindgen::JsValue::from_str("triggerKatex"))
            {
                if func.is_function() {
                    let func: js_sys::Function = func.into();
                    let _ = func.call0(&window);
                }
            }
        }
    });

    view! {
        <Nav />
        <main>
            {move || match page.get() {
                Page::Home         => view! { <HomePage         /> }.into_any(),
                Page::Architecture => view! { <ArchitecturePage /> }.into_any(),
                Page::Sponsor      => view! { <SponsorPage      /> }.into_any(),
                Page::Bench        => view! { <BenchPage        /> }.into_any(),
            }}
        </main>
        <Footer />
    }
}
