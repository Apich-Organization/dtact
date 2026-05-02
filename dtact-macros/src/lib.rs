use proc_macro::TokenStream;
use quote::quote;
use syn::{
    FnArg, ItemFn, Lit, Meta, Token, parse::Parse, parse::ParseStream, parse_macro_input,
    punctuated::Punctuated,
};

struct TaskArgs {
    priority: String,
    affinity: String,
    kind: String,
    stack: String,
    switcher: String,
}

impl Parse for TaskArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let vars = Punctuated::<Meta, Token![,]>::parse_terminated(input)?;
        let mut priority = "Normal".to_string();
        let mut affinity = "SameCore".to_string();
        let mut kind = "Compute".to_string();
        let mut stack = "2M".to_string();
        let mut switcher = "CrossThreadFloat".to_string();

        for var in vars {
            if let Meta::NameValue(nv) = var {
                if nv.path.is_ident("priority") {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: Lit::Str(s), ..
                    }) = nv.value
                    {
                        priority = s.value();
                    }
                } else if nv.path.is_ident("affinity") {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: Lit::Str(s), ..
                    }) = nv.value
                    {
                        affinity = s.value();
                    }
                } else if nv.path.is_ident("kind") {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: Lit::Str(s), ..
                    }) = nv.value
                    {
                        kind = s.value();
                    }
                } else if nv.path.is_ident("stack") {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: Lit::Str(s), ..
                    }) = nv.value
                    {
                        stack = s.value();
                    }
                } else if nv.path.is_ident("switcher")
                    && let syn::Expr::Lit(syn::ExprLit {
                        lit: Lit::Str(s), ..
                    }) = nv.value
                {
                    switcher = s.value();
                }
            }
        }

        Ok(TaskArgs {
            priority,
            affinity,
            kind,
            stack,
            switcher,
        })
    }
}

#[proc_macro_attribute]
pub fn task(args: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as TaskArgs);
    let input = parse_macro_input!(item as ItemFn);

    let fn_name = &input.sig.ident;
    let priority = &args.priority;
    let affinity = &args.affinity;
    let kind = &args.kind;
    let stack = &args.stack;

    let metadata_mod = syn::Ident::new(&format!("dtact_metadata_{}", fn_name), fn_name.span());
    let priority_ident = syn::Ident::new(priority, fn_name.span());
    let affinity_ident = syn::Ident::new(affinity, fn_name.span());
    let kind_ident = syn::Ident::new(kind, fn_name.span());
    let switcher = &args.switcher;
    let switcher_ident = syn::Ident::new(switcher, fn_name.span());

    let expanded = quote! {
        #input

        pub mod #metadata_mod {
            pub const PRIORITY: dtact::Priority = dtact::Priority::#priority_ident;
            pub const AFFINITY: dtact::topology::Affinity = dtact::topology::Affinity::#affinity_ident;
            pub const KIND: dtact::WorkloadKind = dtact::WorkloadKind::#kind_ident;
            pub const STACK_SIZE: &'static str = #stack;
            pub type SWITCHER = dtact::#switcher_ident;
        }
    };

    TokenStream::from(expanded)
}

#[proc_macro_attribute]
pub fn export_async(_args: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let fn_name = &input.sig.ident;
    let wrapper_name = syn::Ident::new(&format!("dtact_export_{}", fn_name), fn_name.span());

    let mut c_params = Vec::new();
    let mut call_args = Vec::new();

    for input in &input.sig.inputs {
        if let FnArg::Typed(pat_type) = input {
            let pat = &pat_type.pat;
            let ty = &pat_type.ty;
            c_params.push(quote! { #pat: #ty });
            call_args.push(quote! { #pat });
        } else {
            panic!("export_async does not support 'self' parameters");
        }
    }

    let expanded = quote! {
        #input

        #[unsafe(no_mangle)]
        pub extern "C" fn #wrapper_name(#(#c_params),*) -> dtact::dtact_handle_t {
            dtact::spawn(#fn_name(#(#call_args),*))
        }
    };

    TokenStream::from(expanded)
}

#[proc_macro_attribute]
pub fn export_fiber(_args: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let fn_name = &input.sig.ident;
    let wrapper_name = syn::Ident::new(&format!("dtact_export_fiber_{}", fn_name), fn_name.span());

    let mut c_params = Vec::new();
    let mut call_args = Vec::new();

    for input in &input.sig.inputs {
        if let FnArg::Typed(pat_type) = input {
            let pat = &pat_type.pat;
            let ty = &pat_type.ty;
            c_params.push(quote! { #pat: #ty });
            call_args.push(quote! { #pat });
        } else {
            panic!("export_fiber does not support 'self' parameters");
        }
    }

    let expanded = quote! {
        #input

        #[unsafe(no_mangle)]
        pub extern "C" fn #wrapper_name(#(#c_params),*) -> dtact::dtact_handle_t {
            dtact::api::fiber::spawn_with_stack("2M", move || {
                #fn_name(#(#call_args),*);
            })
        }
    };

    TokenStream::from(expanded)
}

struct InitArgs {
    topology: String,
    safety: String,
    workers: usize,
    capacity: u32,
    stack: usize,
    numa: usize,
}

impl Parse for InitArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let vars = Punctuated::<Meta, Token![,]>::parse_terminated(input)?;
        let mut topology = "P2PMesh".to_string();
        let mut safety = "Safety1".to_string();
        let mut workers = 0;
        let mut capacity = 16384;
        let mut stack = 2 * 1024 * 1024;
        let mut numa = 0;

        for var in vars {
            if let Meta::NameValue(nv) = var {
                if nv.path.is_ident("topology") {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: Lit::Str(s), ..
                    }) = &nv.value
                    {
                        topology = s.value();
                    }
                } else if nv.path.is_ident("safety") {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: Lit::Str(s), ..
                    }) = &nv.value
                    {
                        safety = s.value();
                    }
                } else if nv.path.is_ident("workers") {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: Lit::Int(i), ..
                    }) = &nv.value
                    {
                        workers = i.base10_parse()?;
                    }
                } else if nv.path.is_ident("capacity") {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: Lit::Int(i), ..
                    }) = &nv.value
                    {
                        capacity = i.base10_parse()?;
                    }
                } else if nv.path.is_ident("stack") {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: Lit::Int(i), ..
                    }) = &nv.value
                    {
                        stack = i.base10_parse()?;
                    }
                } else if nv.path.is_ident("numa")
                    && let syn::Expr::Lit(syn::ExprLit {
                        lit: Lit::Int(i), ..
                    }) = &nv.value
                {
                    numa = i.base10_parse()?;
                }
            }
        }
        Ok(InitArgs {
            topology,
            safety,
            workers,
            capacity,
            stack,
            numa,
        })
    }
}

#[proc_macro_attribute]
pub fn dtact_init(args: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as InitArgs);
    let input = parse_macro_input!(item as ItemFn);

    let topology = &args.topology;
    let safety = &args.safety;
    let workers = args.workers;
    let capacity = args.capacity;
    let stack = args.stack;
    let numa = args.numa;

    let topology_ident = syn::Ident::new(topology, input.sig.ident.span());
    let safety_ident = syn::Ident::new(safety, input.sig.ident.span());

    let expanded = quote! {
        #input

        const _: () = {
            #[unsafe(no_mangle)]
            extern "C" fn dtact_autostart() {
                dtact::GLOBAL_RUNTIME.get_or_init(|| {
                    let mut workers_count = #workers;
                    if workers_count == 0 {
                        workers_count = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
                    }

                    let scheduler = dtact::dta_scheduler::DtaScheduler::new(
                        workers_count,
                        dtact::dta_scheduler::TopologyMode::#topology_ident
                    );

                    let pool = dtact::memory_management::ContextPool::new(
                        #capacity,
                        #stack,
                        dtact::memory_management::SafetyLevel::#safety_ident,
                        #numa
                    ).expect("DTA-V3 Hardware Initialization Failed");

                    dtact::Runtime { scheduler, pool }
                });
            }
        };
    };

    TokenStream::from(expanded)
}
